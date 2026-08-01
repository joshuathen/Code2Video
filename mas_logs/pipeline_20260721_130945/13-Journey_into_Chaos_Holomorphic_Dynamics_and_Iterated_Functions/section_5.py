from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # Retrieve content from shared state
        title_text = "The Map of All Maps: The Mandelbrot Set"
        lecture_lines = [
            "We iterate the function z squared plus c.",
            "The Mandelbrot set maps every possible value of c.",
            "Points inside represent parameters with connected Julia sets.",
            "Each tiny bulb corresponds to a unique fractal world.",
            "It is the ultimate dictionary of complex iteration."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define custom colors as per visual guidelines
        COLOR_FORMULA = "#FFFFFF"
        COLOR_MANDELBROT = "#FF00FF"
        COLOR_PLANE = BLUE_D
        COLOR_SWEEP = WHITE
        COLOR_THUMB = ORANGE
        COLOR_MINI = PINK

        # === Animation for Lecture Line 1 ===
        # Set z_0 = 0 in #FFFFFF and show the formula z_{n+1} = z_n^2 + c 
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg].
        self.lecture[0].set_color(COLOR_FORMULA)
        
        formula = MathTex("z_{n+1} = z_n^2 + c", color=COLOR_FORMULA)
        z0_label = MathTex("z_0 = 0", color=COLOR_FORMULA)
        
        # Integrate Asset per Issue 21
        dictionary_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg")
        dictionary_asset.set_color(COLOR_FORMULA).scale(0.4)
        
        formula_group = VGroup(formula, z0_label, dictionary_asset).arrange(DOWN, buff=0.3)
        
        # Position in top-right area per Issue 34
        self.place_in_area(formula_group, "B3", "B5", scale_factor=0.8)
        
        self.play(Write(formula), run_time=1.5)
        self.play(FadeIn(z0_label, shift=UP*0.2), run_time=1)
        self.play(FadeIn(dictionary_asset, shift=LEFT*0.2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Sweep the complex plane, coloring points that don't escape in black (#000000).
        self.lecture[1].set_color(YELLOW)
        
        # Initialize a coordinate system (NumberPlane)
        plane = NumberPlane(
            x_range=[-2.5, 1.0, 1],
            y_range=[-1.2, 1.2, 1],
            background_line_style={"stroke_color": COLOR_PLANE, "stroke_opacity": 0.3},
            axis_config={"stroke_opacity": 0.5}
        )
        # Position plane in the central/lower right area per Issue 33
        self.place_in_area(plane, "C3", "F6", scale_factor=0.7)
        
        # Visual sweep line effect
        sweep_line = Line(
            plane.get_left(), plane.get_right(), 
            color=COLOR_SWEEP, stroke_width=2, stroke_opacity=0.6
        )
        sweep_line.set_y(plane.get_top()[1])
        
        self.play(Create(plane), run_time=1.5)
        # Linear sweep across the plane's vertical range
        self.play(
            sweep_line.animate.move_to(plane.get_bottom()),
            run_time=2.5,
            rate_func=linear
        )
        self.play(FadeOut(sweep_line))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Reveal the iconic Mandelbrot bulb and cardioid shape with a #FF00FF outline.
        self.lecture[2].set_color(COLOR_MANDELBROT)
        
        # Geometric approximation of the Mandelbrot cardioid
        # c(t) = 1/4 * (2*exp(it) - exp(i2t))
        m_cardioid = ParametricFunction(
            lambda t: plane.coords_to_point(
                0.25 * (2 * np.cos(t) - np.cos(2 * t)),
                0.25 * (2 * np.sin(t) - np.sin(2 * t))
            ),
            t_range=[0, TAU],
            color=COLOR_MANDELBROT,
            stroke_width=3
        )
        
        # Geometric approximation of the period-2 bulb (Circle at -1, radius 1/4)
        bulb_center = plane.coords_to_point(-1, 0)
        # Calculate radius in screen units based on plane scale
        # Note: plane.get_x_unit_size() is the distance between coordinates -1 and 0 in screen space.
        bulb_radius = 0.25 * (plane.get_x_unit_size())
        m_bulb = Circle(radius=bulb_radius, color=COLOR_MANDELBROT, stroke_width=3)
        m_bulb.move_to(bulb_center)
        
        m_set_outline = VGroup(m_cardioid, m_bulb)
        
        # Solid fill to denote points inside the set
        # Using a slight opacity to keep the grid visible if needed, but storyboard says black.
        m_set_filled = m_set_outline.copy().set_fill(BLACK, opacity=1).set_stroke(width=1)
        
        self.play(Create(m_set_outline), run_time=2)
        self.play(FadeIn(m_set_filled))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Small thumbnails of different Julia sets appear around the main set.
        self.lecture[3].set_color(COLOR_THUMB)
        
        # Representative points on the boundary for Julia set samples
        julia_locs = [
            plane.coords_to_point(-0.1, 0.7),
            plane.coords_to_point(-0.8, 0.15),
            plane.coords_to_point(0.28, 0.45),
            plane.coords_to_point(-1.25, 0)
        ]
        
        thumbnails = VGroup()
        for i, pt in enumerate(julia_locs):
            # Complex-looking polygon shapes representing localized fractal structures
            sides = 10 + i * 2
            thumb = RegularPolygon(n=sides, color=COLOR_THUMB, fill_opacity=0.5).scale(0.1)
            thumb.move_to(pt)
            thumbnails.add(thumb)
            
        self.play(
            LaggedStart(
                *[FadeIn(t, scale=1.3) for t in thumbnails],
                lag_ratio=0.4
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Scale the entire view to show the 'mini-mandelbrots' along the filaments.
        self.lecture[4].set_color(COLOR_MINI)
        
        # Create a "mini-mandelbrot" copy at a filament point (e.g., the tail)
        mini_m_loc = plane.coords_to_point(-1.75, 0)
        mini_m = m_set_outline.copy().scale(0.12).set_color(COLOR_MINI)
        mini_m.move_to(mini_m_loc)
        
        # Combine all visual elements for a synchronized zoom
        all_elements = VGroup(plane, m_set_outline, m_set_filled, thumbnails, mini_m)
        
        self.play(FadeIn(mini_m))
        # Zoom centered on the mini-mandelbrot location
        # [L015] Note: The zoom might cause overlap with lecture lines if not handled. 
        # The scale factor is 5, but let's ensure it doesn't cross the left boundary.
        # The plane is at C3-F6, which is x >= 2.5 (0.5 + 2*1). 
        # Left side lecture ends around x=0. 
        # We need to be careful with scaling about_point.
        
        self.play(
            all_elements.animate.scale(5, about_point=mini_m_loc),
            run_time=3
        )
        # [L004] Use Indicate to highlight the recursive structure
        self.play(Indicate(mini_m, color=COLOR_MINI, scale_factor=1.2))
        self.wait(2)
