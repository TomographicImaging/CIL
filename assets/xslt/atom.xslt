<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:atom="http://www.w3.org/2005/Atom">
<xsl:output method="html" encoding="utf-8" />
<xsl:template match="/atom:feed">
	<xsl:text disable-output-escaping="yes">&lt;!DOCTYPE html &gt;</xsl:text>
	<html>
	<head>
		<xsl:text disable-output-escaping="yes"><![CDATA[
		<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Atom Feed (Styled)</title>

    <link rel="stylesheet" type="text/css" href="/CIL/assets/css/styles_feeling_responsive.css">

  

	<script src="/CIL/assets/js/modernizr.min.js"></script>

	<script src="https://ajax.googleapis.com/ajax/libs/webfont/1.5.18/webfont.js"></script>
	<script>
		WebFont.load({
			google: {
				families: [ 'Lato:400,700,400italic:latin', 'Volkhov::latin' ]
			}
		});
	</script>

	<noscript>
		<link href='http://fonts.googleapis.com/css?family=Lato:400,700,400italic%7CVolkhov' rel='stylesheet' type='text/css'>
	</noscript>


	<!-- Search Engine Optimization -->
	<meta name="description" content="CIL is a versatile python framework for tomographic imaging.

CCPi is funded by the EPSRC grant EP/T026677/1.

CCPi acknowledges previous funding support by the EPSRC grants EP/J010456/1
and EP/M022498/1.

The CCPi Flagship &quot;A Reconstruction Toolkit for Multichannel CT&quot; was funded by the
EPSRC grant EP/P02226X/1.">
	
	
	
	
	
	<link rel="canonical" href="/CIL/assets/xslt/atom.xslt">


	<!-- Facebook Open Graph -->
	<meta property="og:title" content="Atom Feed (Styled)">
	<meta property="og:description" content="CIL is a versatile python framework for tomographic imaging.

CCPi is funded by the EPSRC grant EP/T026677/1.

CCPi acknowledges previous funding support by the EPSRC grants EP/J010456/1
and EP/M022498/1.

The CCPi Flagship &quot;A Reconstruction Toolkit for Multichannel CT&quot; was funded by the
EPSRC grant EP/P02226X/1.">
	<meta property="og:url" content="/CIL/assets/xslt/atom.xslt">
	<meta property="og:locale" content="en_EN">
	<meta property="og:type" content="website">
	<meta property="og:site_name" content="CIL">
	
	


	

	<link type="text/plain" rel="author" href="/CIL/humans.txt">

	

	<link rel="icon" sizes="32x32" href="https://ccpi.ac.uk/wp-content/uploads/2022/11/cropped-CCPi_Logo_Icon_Only-32x32.png">


	

	


		]]></xsl:text>
	</head>
	<body id="top-of-page">
		<xsl:text disable-output-escaping="yes"><![CDATA[
		
<div id="navigation" class="sticky">
  <nav class="top-bar" role="navigation" data-topbar data-options="scrolltop: false">
    <ul class="title-area">
      <li class="name">
      <h1 class="hide-for-large-up"><a href="/CIL" class="icon-tree"> CIL</a></h1>
    </li>
       <!-- Remove the class "menu-icon" to get rid of menu icon. Take out "Menu" to just have icon alone -->
      <li class="toggle-topbar toggle-topbar-click menu-icon"><a><span>Nav</span></a></li>
    </ul>
    <section class="top-bar-section">

      <ul class="left">
        

              

          
          

            
            
              <li><a  href="/CIL/">Home</a></li>
              <li class="divider"></li>

            
            
          
        

              

          
          

            
            
              <li><a  href="/CIL/about/">About</a></li>
              <li class="divider"></li>

            
            
          
        

              

          
          

            
            

              <li class="has-dropdown">
                <a  href="/CIL/nightly/">Documentation</a>

                  <ul class="dropdown">
                    

                      

                      <li><a  href="/CIL/nightly/">Documentation</a></li>
                    

                      

                      <li><a  href="https://github.com/TomographicImaging/CIL/releases" target="_blank">Changelog</a></li>
                    

                      

                      <li><a  href="/CIL/nightly/developer_guide/">Developer Guide</a></li>
                    
                  </ul>

              </li>
              <li class="divider"></li>
            
          
        

              

          
          

            
            
              <li><a  href="https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb" target="_blank">Launch Demos</a></li>
              <li class="divider"></li>

            
            
          
        

              

          
          
        

              

          
          
        
        
      </ul>
      
      

      <ul class="right">
        

              



          
          
        

              



          
          
        

              



          
          
        

              



          
          
        

              



          
          
            
            

              <li class="divider"></li>
              <li class="has-dropdown">
                <a  href="https://github.com/TomographicImaging" target="_blank">GitHub</a>

                  <ul class="dropdown">
                    

                      

                      <li><a  href="https://github.com/TomographicImaging/CIL" target="_blank">CIL</a></li>
                    

                      

                      <li><a  href="https://github.com/TomographicImaging/CIL-Demos" target="_blank">CIL-Demos</a></li>
                    
                  </ul>

              </li>
            
          
        

              



          
          
            
            

              <li class="divider"></li>
              <li class="has-dropdown">
                <a  href="/CIL/nightly/#contacts">Contact</a>

                  <ul class="dropdown">
                    

                      

                      <li><a  href="https://discord.gg/9NTWu9MEGq" target="_blank">Discord</a></li>
                    

                      

                      <li><a  href="https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=CCPI-MEMBERS" target="_blank">Member Mailing List</a></li>
                    

                      

                      <li><a  href="https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=CCPI-DEVEL" target="_blank">Developer Mailing List</a></li>
                    

                      

                      <li><a  href="https://zenodo.org/communities/ccpi/about" target="_blank">Zenodo Community</a></li>
                    

                      

                      <li><a  href="https://www.linkedin.com/groups/9090791/" target="_blank">LinkedIn</a></li>
                    
                  </ul>

              </li>
            
          
        
        
      </ul>
     
    </section>
  </nav>
</div><!-- /#navigation -->

		

<div id="masthead-no-image-header">
	<div class="row">
		<div class="small-12 columns">
			<a id="logo" href="/CIL/" title="CIL – The Core Imaging Library">
				<img src="/CIL/assets/img/https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg" alt="CIL – The Core Imaging Library">
			</a>
		</div><!-- /.small-12.columns -->
	</div><!-- /.row -->
</div><!-- /#masthead -->








		


<div class="alert-box warning text-center"><p>This <a href="https://en.wikipedia.org/wiki/RSS" target="_blank">Atom feed</a> is meant to be used by <a href="https://en.wikipedia.org/wiki/Template:Aggregators" target="_blank">RSS reader applications and websites</a>.</p>
</div>



		]]></xsl:text>
		<header class="t30 row">
	<p class="subheadline"><xsl:value-of select="atom:subtitle" disable-output-escaping="yes" /></p>
	<h1>
		<xsl:element name="a">
			<xsl:attribute name="href">
				<xsl:value-of select="atom:id" />
			</xsl:attribute>
			<xsl:value-of select="atom:title" />
		</xsl:element>
	</h1>
</header>
<ul class="accordion row" data-accordion="">
	<xsl:for-each select="atom:entry">
		<li class="accordion-navigation">
			<xsl:variable name="slug-id">
				<xsl:call-template name="slugify">
					<xsl:with-param name="text" select="atom:id" />
				</xsl:call-template>
			</xsl:variable>
			<xsl:element name="a">
				<xsl:attribute name="href"><xsl:value-of select="concat('#', $slug-id)"/></xsl:attribute>
				<xsl:value-of select="atom:title"/>
				<br/>
				<small><xsl:value-of select="atom:updated"/></small>
			</xsl:element>
			<xsl:element name="div">
				<xsl:attribute name="id"><xsl:value-of select="$slug-id"/></xsl:attribute>
				<xsl:attribute name="class">content</xsl:attribute>
				<h1>
					<xsl:element name="a">
						<xsl:attribute name="href"><xsl:value-of select="atom:id"/></xsl:attribute>
						<xsl:value-of select="atom:title"/>
					</xsl:element>
				</h1>
				<xsl:value-of select="atom:content" disable-output-escaping="yes" />
			</xsl:element>
		</li>
	</xsl:for-each>
</ul>

		<xsl:text disable-output-escaping="yes"><![CDATA[
		    <!-- from https://github.com/Phlow/feeling-responsive/raw/gh-pages/_includes/_footer.html -->
    <div id="up-to-top" class="row">
      <div class="small-12 columns" style="text-align: right;">
        <a class="iconfont" href="#top-of-page">&#xf108;</a>
      </div><!-- /.small-12.columns -->
    </div><!-- /.row -->


    <footer id="footer-content" class="bg-grau">
      <div id="footer">
        <div class="row">
          <div class="medium-6 large-5 columns">
            <h5 class="shadow-black">About This Site</h5>

            <p class="shadow-black">
              CIL is a versatile python framework for tomographic imaging.
<br/>
CCPi is funded by the EPSRC grant <a href="https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/T026677/1">EP/T026677/1</a>.
<br/>
CCPi acknowledges previous funding support by the EPSRC grants <a href="https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/J010456/1">EP/J010456/1</a>
and <a href="https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/M022498/1">EP/M022498/1</a>.
<br/>
The CCPi Flagship "A Reconstruction Toolkit for Multichannel CT" was funded by the
EPSRC grant <a href="https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/P02226X/1">EP/P02226X/1</a>.
<br/>

              <a href="/CIL/about/">More ›</a>
            </p>
          </div><!-- /.large-6.columns -->


          <div class="small-6 medium-3 large-3 large-offset-1 columns">
            
              
                <h5 class="shadow-black">Contact</h5>
              
            
              
            
              
            
              
            
              
            
              
            

              <ul class="no-bullet shadow-black">
              
                
                  <li >
                    <a href=""  title=""></a>
                  </li>
              
                
                  <li >
                    <a href="https://discord.gg/9NTWu9MEGq" target="_blank"  title="">Discord</a>
                  </li>
              
                
                  <li >
                    <a href="https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=CCPI-MEMBERS" target="_blank"  title="Member Mailing List">Member List</a>
                  </li>
              
                
                  <li >
                    <a href="https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=CCPI-DEVEL" target="_blank"  title="Developer Mailing List">Developer List</a>
                  </li>
              
                
                  <li >
                    <a href="https://zenodo.org/communities/ccpi/about" target="_blank"  title="Zenodo Community">Zenodo</a>
                  </li>
              
                
                  <li >
                    <a href="https://www.linkedin.com/groups/9090791/" target="_blank"  title="LinkedIn Group">LinkedIn</a>
                  </li>
              
              </ul>
          </div><!-- /.large-4.columns -->


          <div class="small-6 medium-3 large-3 columns">
            
              
                <h5 class="shadow-black">Thanks</h5>
              
            
              
            
              
            
              
            
              
            
              
            

            <ul class="no-bullet shadow-black">
            
              
                <li >
                  <a href=""  title=""></a>
                </li>
            
              
                <li >
                  <a href="https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx" target="_blank"  title="Computational Science Centre for Research Communities">CoSeC</a>
                </li>
            
              
                <li >
                  <a href="https://www.ukri.org" target="_blank"  title="UK Research and Innovation">UKRI</a>
                </li>
            
              
                <li >
                  <a href="https://stfc.ac.uk" target="_blank"  title="Science and Technology Facilities Council">STFC</a>
                </li>
            
              
                <li >
                  <a href="https://epsrc.ac.uk" target="_blank"  title="Engineering and Physical Sciences Research Council">EPSRC</a>
                </li>
            
              
                <li >
                  <a href="https://www.manchester.ac.uk" target="_blank"  title="University of Manchester">Manchester</a>
                </li>
            
            </ul>
          </div><!-- /.large-3.columns -->
        </div><!-- /.row -->

      </div><!-- /#footer -->


      <div id="subfooter">
        <nav class="row">
          <p style="text-align: center;"><a href="https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx" target="_blank" title="Computational Science Centre for Research Communities"><img loading="lazy" width="85" height="64" src="https://ccpi.ac.uk/wp-content/uploads/2022/10/CoSec_Transparent-002.png"/></a><a href="https://stfc.ac.uk" target="_blank" title="Science and Technology Facilities Council"><img loading="lazy" width="249" height="64" src="https://ccpi.ac.uk/wp-content/uploads/2022/10/STFC_White_Text-2.png"/></a><a href="https://epsrc.ac.uk" target="_blank" title="Engineering and Physical Sciences Research Council"><img loading="lazy" width="256" height="64" src="https://ccpi.ac.uk/wp-content/uploads/2022/10/epsrc.png"/></a><a href="https://www.manchester.ac.uk" target="_blank" title="University of Manchester"><img loading="lazy" width="151" height="64" src="https://ccpi.ac.uk/wp-content/uploads/2022/10/TAB_col_background_manc-300x127.png"/></a><br/>Copyright &copy; UKRI STFC
          </p>
        </nav>
        <nav class="row">
          <section id="subfooter-left" class="small-12 medium-6 columns credits">
            
          </section>

          <section id="subfooter-right" class="small-12 medium-6 columns">
            <ul class="inline-list social-icons">
            
            </ul>
          </section>
        </nav>
      </div><!-- /#subfooter -->
    </footer>

		


<script src="/CIL/assets/js/javascript.min.js"></script>












		]]></xsl:text>
	</body>
	</html>
</xsl:template>
<xsl:template name="slugify">
	<xsl:param name="text" select="''" />
	<xsl:variable name="dodgyChars" select="' ,.#_-!?*:;=+|&amp;/\\'" />
	<xsl:variable name="replacementChar" select="'-----------------'" />
	<xsl:variable name="lowercase" select="'abcdefghijklmnopqrstuvwxyz'" />
	<xsl:variable name="uppercase" select="'ABCDEFGHIJKLMNOPQRSTUVWXYZ'" />
	<xsl:variable name="lowercased"><xsl:value-of select="translate( $text, $uppercase, $lowercase )" /></xsl:variable>
	<xsl:variable name="escaped"><xsl:value-of select="translate( $lowercased, $dodgyChars, $replacementChar )" /></xsl:variable>
	<xsl:value-of select="$escaped" />
</xsl:template>
</xsl:stylesheet>
