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

class Section4Scene(TeachingScene):
    def construct(self):
        # Fetching context from storyboard/outline
        title = "The Geometry of the Intersection"
        lines = [
            "Evidence has occurred, so only shaded areas remain possible.",
            "We focus on the intersection of hypothesis and evidence.",
            "These two rectangles represent our new world of possibilities."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_BLUE = "#58C4DD"
        COLOR_YELLOW = "#F9F295"
        COLOR_FADED = "#333333"
        
        # 1. Unit Square Background (Area A2 to E5)
        # This provides a 3x4 grid-unit frame.
        unit_square = Rectangle(width=3.0, height=4.0, color=WHITE, stroke_width=2)
        self.place_in_area(unit_square, "A2", "E5")
        
        # 2. Blue Rectangle (Represents P(H and E))
        # Spans rows B, C, D (height 2) and cols 2, 3 (width 1)
        blue_rect = Rectangle(
            width=1.0, height=2.0, 
            fill_color=COLOR_BLUE, fill_opacity=0.8, stroke_color=WHITE
        )
        self.place_in_area(blue_rect, "B2", "D3")
        
        # 3. Yellow Rectangle (Represents P(not H and E))
        # Occupies cell B4.
        yellow_rect = Rectangle(
            width=1.0, height=1.0, 
            fill_color=COLOR_YELLOW, fill_opacity=0.8, stroke_color=WHITE
        )
        self.place_at_grid(yellow_rect, "B4")
        
        # Labels
        # Resolve Issues 30 and 31 by scaling labels to 0.6 to avoid boundary overlap
        blue_label = MathTex("P(H \\cap E)", font_size=24, color=COLOR_BLUE)
        self.place_at_grid(blue_label, "E2", scale_factor=0.6)
        
        yellow_label = MathTex("P(\\neg H \\cap E)", font_size=24, color=COLOR_YELLOW)
        self.place_at_grid(yellow_label, "A4", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_BLUE)
        self.add(unit_square)
        self.play(FadeIn(blue_rect), FadeIn(yellow_rect))
        self.wait(1)
        
        # Fade out unshaded areas by dimming the main square
        self.play(unit_square.animate.set_stroke(color=COLOR_FADED, opacity=0.5))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_BLUE)
        
        # Highlight intersection with labels
        self.play(
            blue_rect.animate.set_stroke(width=6),
            yellow_rect.animate.set_stroke(width=6),
            Write(blue_label),
            Write(yellow_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_YELLOW)
        
        # Resolve Issue 29: total_ev_label alignment and scaling
        total_ev_label = Text("Total Probability of Evidence P(E)", font_size=20, color=WHITE)
        self.place_in_area(total_ev_label, 'F2', 'F5', scale_factor=0.6)
        
        # Pulse both to show total evidence P(E)
        self.play(
            blue_rect.animate.scale(1.1),
            yellow_rect.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=0.8
        )
        
        self.play(Write(total_ev_label))
        self.wait(2)
