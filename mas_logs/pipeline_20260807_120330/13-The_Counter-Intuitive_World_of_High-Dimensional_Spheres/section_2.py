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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the n-Sphere", [
            "Sⁿ is the surface; Bⁿ is the solid ball.",
            "An n-sphere lives in n+1 dimensional space.",
            "Each new dimension adds a new axis of symmetry."
        ])
        
        # Colors
        s_color = "#FFFFFF"
        b_fill_color = "#808080"
        highlight_color = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # Script: "Sⁿ is the surface; Bⁿ is the solid ball."
        # Visual: Show a circle S¹ and a sphere S². Labels #FFFFFF at top.
        self.play(self.lecture[0].animate.set_color(highlight_color))
        
        # S1 (Circle) - Shifted to column 3 (Issue 25)
        s1 = Circle(radius=0.45, color=s_color)
        self.place_at_grid(s1, "C3")
        s1_label = MathTex("S^1", color=s_color)
        self.place_at_grid(s1_label, "B3", scale_factor=0.8)
        
        # S2 (Sphere representation) - Shifted to column 6 (Issue 26)
        s2_outline = Circle(radius=0.45, color=s_color)
        s2_equator = Ellipse(width=0.9, height=0.25, color=s_color).set_stroke(opacity=0.5)
        s2 = VGroup(s2_outline, s2_equator)
        self.place_at_grid(s2, "C6")
        s2_label = MathTex("S^2", color=s_color)
        self.place_at_grid(s2_label, "B6", scale_factor=0.8)
        
        self.play(
            Create(s1), Write(s1_label),
            Create(s2), Write(s2_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Script: "An n-sphere lives in n+1 dimensional space."
        # Visual: Label S¹ as 'In 2D space' and S² as 'In 3D space'.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE)
        )
        
        # Updated space labels using place_in_area (Issue 27)
        s1_space_label = Text("In 2D space", font_size=18, color=WHITE)
        self.place_in_area(s1_space_label, 'D3', 'D3', scale_factor=0.8)
        
        s2_space_label = Text("In 3D space", font_size=18, color=WHITE)
        self.place_in_area(s2_space_label, 'D6', 'D6', scale_factor=0.8)
        
        self.play(FadeIn(s1_space_label), FadeIn(s2_space_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Script: "Each new dimension adds a new axis of symmetry."
        # Visual: Fill S¹ and S² with #808080 to represent B¹ and B². Highlight labels.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )
        
        # B1 and B2 fills - Grid updates (Issue 25 & 26)
        b1_fill = Circle(radius=0.45, fill_opacity=0.6, fill_color=b_fill_color, stroke_width=0)
        self.place_at_grid(b1_fill, "C3")
        
        b2_fill = Circle(radius=0.45, fill_opacity=0.6, fill_color=b_fill_color, stroke_width=0)
        self.place_at_grid(b2_fill, "C6")
        
        # New labels to emphasize B^n
        b1_label = MathTex("B^1", color=highlight_color)
        self.place_at_grid(b1_label, "A3", scale_factor=0.8)
        
        b2_label = MathTex("B^2", color=highlight_color)
        self.place_at_grid(b2_label, "A6", scale_factor=0.8)
        
        # Symmetry axes
        axis1 = Line(s1.get_left(), s1.get_right(), color=GREEN, stroke_width=2)
        axis2 = Line(s1.get_top(), s1.get_bottom(), color=GREEN, stroke_width=2)
        axis3 = Line(s2.get_top(), s2.get_bottom(), color=GREEN, stroke_width=2)
        
        self.play(
            FadeIn(b1_fill), FadeIn(b2_fill),
            Write(b1_label), Write(b2_label),
            s1_label.animate.set_color(highlight_color),
            s2_label.animate.set_color(highlight_color),
            Create(axis1), Create(axis2), Create(axis3)
        )
        self.wait(3)
