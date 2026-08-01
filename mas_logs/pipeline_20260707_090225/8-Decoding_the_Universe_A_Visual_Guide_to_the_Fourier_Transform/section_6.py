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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            'This tool powers image compression and medical scans.',
            'It reveals secrets hidden in signals from deep space.',
            'Everything in our universe is built from simple waves.'
        ]
        self.setup_layout("Summary: The Universal Language", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Icons representing a jar, a digital photo, and a medical scan appear.
        line1_color = "#ADD8E6"
        self.play(self.lecture[0].animate.set_color(line1_color))
        
        # Jar Icon (Asset integrated from Issue 28)
        jar_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/jar.svg", color=line1_color)
        self.place_at_grid(jar_icon, "B2", scale_factor=0.7) # Position/Scale updated per Issue 40
        
        # Photo Icon (Digital Image)
        photo_frame = Square(side_length=1.0, color=line1_color)
        photo_inner = Rectangle(height=0.6, width=0.8, color=line1_color).move_to(photo_frame.get_center())
        photo_dot = Dot(color=line1_color).scale(0.5).move_to(photo_frame.get_corner(UL) + RIGHT*0.2 + DOWN*0.2)
        photo_icon = VGroup(photo_frame, photo_inner, photo_dot)
        self.place_at_grid(photo_icon, "B3", scale_factor=0.7) # Position/Scale updated per Issue 40
        
        # Medical Scan Icon (MRI/Brain scan)
        scan_outer = Circle(radius=0.5, color=line1_color)
        scan_inner = Ellipse(width=0.4, height=0.6, color=line1_color).move_to(scan_outer.get_center())
        scan_icon = VGroup(scan_outer, scan_inner)
        self.place_at_grid(scan_icon, "B4", scale_factor=0.7) # Position/Scale updated per Issue 40
        
        self.play(
            FadeIn(jar_icon),
            FadeIn(photo_icon),
            FadeIn(scan_icon)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # All icons are connected by glowing white lines to the text 'Fourier Transform' (#FFFFFF).
        line2_color = "#FFFFFF"
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(line2_color)
        )
        
        ft_text = Text("Fourier Transform", font_size=24, color=WHITE)
        self.place_at_grid(ft_text, "C3", scale_factor=0.8) # Position/Scale updated per Issue 41
        
        line_jar = Line(jar_icon.get_bottom(), ft_text.get_top(), color=WHITE, stroke_width=2)
        line_photo = Line(photo_icon.get_bottom(), ft_text.get_top(), color=WHITE, stroke_width=2)
        line_scan = Line(scan_icon.get_bottom(), ft_text.get_top(), color=WHITE, stroke_width=2)
        
        glowing_lines = VGroup(line_jar, line_photo, line_scan)
        
        self.play(Write(ft_text))
        self.play(Create(glowing_lines))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The screen fades to black as a single golden wave (#FFD700) pulses from the center.
        line3_color = "#FFD700"
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(line3_color)
        )
        
        # Group right-side elements and general UI to fade out
        right_side_content = VGroup(jar_icon, photo_icon, scan_icon, ft_text, glowing_lines)
        ui_elements = VGroup(self.title, self.lecture[0], self.lecture[1])
        
        # Golden Wave construction
        golden_wave = FunctionGraph(
            lambda x: 0.7 * np.sin(2 * PI * x / 2),
            x_range=[-3, 3],
            color=line3_color
        )
        # Position updated per Issue 42
        self.place_in_area(golden_wave, "C1", "E6", scale_factor=0.9)
        
        # Transition to black background for the wave pulse
        self.play(
            FadeOut(right_side_content),
            FadeOut(ui_elements),
            run_time=1.5
        )
        
        self.play(Create(golden_wave))
        
        # Pulsing effect (scaling)
        for _ in range(2):
            self.play(golden_wave.animate.scale(1.25), run_time=0.8, rate_func=there_and_back)
        
        self.wait(2)
