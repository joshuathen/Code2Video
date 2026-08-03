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
        # Fetching titles and lines from storyboard
        title = "Prerequisite: Conditional Probability & The Shrinking Universe"
        lines = [
            "Conditional probability looks at events within a restricted universe.",
            "We imagine our sample space shrinking to event B.",
            "We then find the probability of A within B."
        ]
        self.setup_layout(title, lines)

        # Defined Colors
        color_s = "#D3D3D3"
        color_b = "#FFB6C1"
        color_a = "#ADD8E6"
        color_intersect = "#00FF00"
        color_text = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # "Conditional probability looks at events within a restricted universe."
        
        # Rectangle S (Issue 24: Use A1 to E6)
        s_rect = Rectangle(width=5.0, height=4.0, color=color_s, fill_opacity=0.2)
        self.place_in_area(s_rect, "A1", "E6")
        
        # Label S
        s_label = Text("Universal Set (S)", font_size=20, color=color_text)
        self.place_at_grid(s_label, "A3")

        # Region B
        b_circle_initial = Circle(radius=0.7, color=color_b, fill_opacity=0.4)
        self.place_at_grid(b_circle_initial, "D4")
        b_label_init = Text("B", font_size=18, color=color_b)
        self.place_at_grid(b_label_init, "D5")

        self.play(
            self.lecture[0].animate.set_color(color_s),
            Create(s_rect),
            Write(s_label),
            FadeIn(b_circle_initial),
            Write(b_label_init)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We imagine our sample space shrinking to event B."
        
        # B expands to fill S's area (Issue 24: Use A1 to E6)
        b_expanded = Rectangle(width=5.0, height=4.0, color=color_b, fill_opacity=0.2)
        self.place_in_area(b_expanded, "A1", "E6")
        
        b_label = Text("New Universe: B", font_size=20, color=color_b)
        self.place_at_grid(b_label, "A3")

        self.play(
            self.lecture[1].animate.set_color(color_b),
            FadeOut(s_rect),
            FadeOut(s_label),
            FadeOut(b_label_init),
            Transform(b_circle_initial, b_expanded),
            Write(b_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We then find the probability of A within B."
        
        a_circle = Circle(radius=1.0, color=color_a, fill_opacity=0.3)
        self.place_at_grid(a_circle, "C4")
        
        # Intersection Label (Issue 26: Center at C4)
        a_intersect_label = MathTex(r"A \cap B", color=color_intersect).scale(0.8)
        self.place_at_grid(a_intersect_label, "C4")

        # Formula (Issue 25: Use F1 to F6)
        formula = MathTex(
            r"P(A|B) = \frac{\text{Area}(A \cap B)}{\text{Area}(B)}",
            color=color_text
        ).scale(0.9)
        self.place_in_area(formula, "F1", "F6")

        self.play(
            self.lecture[2].animate.set_color(color_a),
            Create(a_circle)
        )
        self.play(
            a_circle.animate.set_fill(color_intersect, opacity=0.6),
            Write(a_intersect_label)
        )
        self.play(
            Write(formula)
        )
        self.wait(3)
