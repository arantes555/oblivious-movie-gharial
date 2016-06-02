from HTMLParser import HTMLParser
import re


# create a subclass and override the handler methods
class ReviewParser(HTMLParser):
    def __init__(self):
        self.title = ''
        self.reviewer = ''
        self.capsule_review = ''
        self.review = ''
        self.review_paragraphs = []
        self.rating = 0
    
        self._inBody = False
        self._inH1 = False
        self._inH3 = False
        self._inReview = False

        self._gotHeader = False
        self._gotFooter = False

        HTMLParser.__init__(self)

    def handle_starttag(self, tag, attrs):
        if tag == 'body':
            self._inBody = True
        elif self._inBody:    
            if tag == 'h1':
                self._inH1 = True
            elif tag == 'h3':
                self._inH3 = True
            elif tag == 'pre' and self._gotHeader:
                self._gotFooter = True
            elif tag == 'p' and self._gotHeader and not self._gotFooter:
                self._inReview = True

    def handle_endtag(self, tag):
        if tag == 'body':
            self._inBody = False
        elif self._inBody:
            if tag == 'h1':
                self._inH1 = False
            if tag == 'h3':
                self._inH3 = False
            elif tag == 'pre' and not self._gotHeader:
                self._gotHeader = True
            elif tag == 'p' and self._gotHeader and not self._gotFooter:
                self._inReview = False

    def handle_data(self, data):
        if self._inBody:
            if self._inH1:
                self.title = data
            elif self._inH3:
                if not data == 'reviewed by':
                    self.reviewer += data
            elif self._inReview:
                if not self.capsule_review:
                    self.capsule_review = data.replace('          Capsule review:  ', '', 1).replace('\n     ', ' ')
                else:
                    self.review += data + '\n'

    def parse_review(self):
        split_rating = re.split('[\\w\\s]+([\\-\\+][0-4]) on the \\-4 to \\+4 scale\\.', self.review)
        self.review = split_rating[0]
        self.rating = split_rating[1]
        return {
            'movie': self.title,
            'reviewer': self.reviewer,
            'review': self.review,
            'capsule_review': self.capsule_review,
            'rating': self.rating
        }

    @staticmethod
    def parse(review_html):
        parser = ReviewParser()
        parser.feed(review_html)
        return parser.parse_review()

myReviewHtml = """
<HTML><HEAD>
<TITLE>Review for Belizaire the Cajun (1986)</TITLE>
<LINK REL="STYLESHEET" TYPE="text/css" HREF="/ramr.css">
</HEAD>
<BODY BGCOLOR="#FFFFFF" TEXT="#000000">
<H1 ALIGN="CENTER" CLASS="title"><A HREF="/Title?0090710">Belizaire the Cajun (1986)</A></H1><H3 ALIGN=CENTER>reviewed by<BR><A HREF="/ReviewsBy?Mark+R.+Leeper">Mark R. Leeper</A></H3><HR WIDTH="40%" SIZE="4">
<PRE>
                             BELIZAIRE THE CAJUN
                       A film review by Mark R. Leeper
                        Copyright 1987 Mark R. Leeper</PRE>
<P>          Capsule review:  Mark Twain-esque tale of murder and
     bigotry set in 1859 in Louisiana.  Armand Assante plays a
     clever Cajun healer trying to protect his cousin and the
     whole Cajun community from local rednecks.  This is an
     engrossing and enjoyable film that tells its story without
     any wrong moves.  Recommended.</P>
<P>     Every once in a while, one of the cable stations runs what they call a
"find."  That is, they run a little-known film that got almost no theatrical
play but is a good film anyway.  Actually, "little-known" may not be quite
accurate, since almost always the cable "finds" have been previously
reviewed--usually favorably--on one or the other of TV's review programs.
Well, I guess in the same sense I can claim BELIZAIRE THE CAJUN as a find of
sorts.  It seems to me that I had heard it recommended but it sounded kind
of artsy and dull.  The last thing I was expecting was a fun film.  It takes
about five minutes to get into the southern and Cajun accents and the
occasional lapsing into French.  (What can I say?  I'm slow!)  After that I
was enjoying the film too much to let any of that get in my way.</P>
<P>     In southwestern Louisiana in 1859 the men of the upper class are
joining together in a vigilante "committee"--the forerunner of the Ku Klux
Klan--to frighten the local Cajun population into leaving the territory.
Not yet threatened and for now perhaps the best-liked of the Cajuns is
Belizaire, a healer, a fast thinker, and a real character.  But Belizaire's
cousin has been given a writ of exile by the vigilantes and Belizaire is
quickly being pulled into the fray.</P>
<P>     In a story worthy of Mark Twain--in fact, reminiscent in some ways of
PUDDIN' HEAD WILSON--we have a story about an ethnic group rarely seen in
films, yet a story that will be meaningful to every minority and should be
enjoyable by just about anyone.</P>
<P>     Admittedly I do not know much about Cajun culture, but a viewer can
generally tell when a film seems authentic and this one seems to have been
very well researched.  Perhaps that is partially due to having Robert Duvall
as creative consultant.  Duvall (who, incidentally, appears in a cameo) has
an extremely good ear for accents and is reputed to be a real stickler for
accuracy.  With a feel of accuracy and a good piece of story-telling,
BELIZAIRE THE CAJUN is a "find" worth looking for at your local video store.
Rate it +2 on the -4 to +4 scale.</P>
<PRE>                                        Mark R. Leeper
                                        ihnp4!mtgzz!leeper
                                        <A HREF="mailto:mtgzz!leeper@rutgers.rutgers.edu">mtgzz!leeper@rutgers.rutgers.edu</A></PRE>
<HR><P CLASS=flush><SMALL>The review above was posted to the
<A HREF="news:rec.arts.movies.reviews">rec.arts.movies.reviews</A> newsgroup (<A HREF="news:de.rec.film.kritiken">de.rec.film.kritiken</A> for German reviews).<BR>
The Internet Movie Database accepts no responsibility for the contents of the
review and has no editorial control. Unless stated otherwise, the copyright
belongs to the author.<BR>
Please direct comments/criticisms of the review to relevant newsgroups.<BR>
Broken URLs inthe reviews are the responsibility of the author.<BR>
The formatting of the review is likely to differ from the original due
to ASCII to HTML conversion.
</SMALL></P>
<P ALIGN=CENTER>Related links: <A HREF="/Reviews/">index of all rec.arts.movies.reviews reviews</A></P>



</P></BODY></HTML>
"""

myReview = ReviewParser.parse(myReviewHtml)

print 'Movie : ', myReview['movie']
print 'Reviewer : ', myReview['reviewer']
print 'Review : ', myReview['review']
print 'Capsule Review :', myReview['capsule_review']
print 'Rating : ', myReview['rating']
