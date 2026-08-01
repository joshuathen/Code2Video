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
        # Initialize the teaching layout
        lecture_lines = [
            'Calculus links slopes and areas in perfect harmony.',
            'Breaking down and building up are two sides.',
            'This bridge connects the two main pillars of calculus.'
        ]
        self.setup_layout("Summary: The Beautiful Symmetry", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Vertical divider for the animation area
        # Grid cols 1-3 (Left) and 4-6 (Right). Middle is at x=3.0.
        divider_top = self.grid["A3"] + RIGHT * 0.5 + UP * 0.5
        divider_bottom = self.grid["F3"] + RIGHT * 0.5 + DOWN * 0.5
        divider = Line(divider_top, divider_bottom, color=WHITE, stroke_width=2)

        # Left/Right labels
        diff_label = Text("Differentiation", font_size=24, color="#FFFFFF")
        int_label = Text("Integration", font_size=24, color="#FFFFFF")
        
        # Position labels using refined areas and scales from Issues 49, 50, 57
        self.place_in_area(diff_label, "B1", "C2", scale_factor=0.8)
        self.place_in_area(int_label, "B5", "C6", scale_factor=0.8)

        self.play(
            Create(divider),
            Write(diff_label),
            Write(int_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Circular arrows connecting Differentiation and Integration
        # Top arrow from Diff to Int
        arrow_top = CurvedArrow(
            diff_label.get_top() + RIGHT * 0.2 + UP * 0.2,
            int_label.get_top() + LEFT * 0.2 + UP * 0.2,
            angle=-TAU/4,
            color="#FFFF00"
        )
        
        # Bottom arrow from Int to Diff
        arrow_bottom = CurvedArrow(
            int_label.get_bottom() + LEFT * 0.2 + DOWN * 0.2,
            diff_label.get_bottom() + RIGHT * 0.2 + DOWN * 0.2,
            angle=-TAU/4,
            color="#FFFF00"
        )

        self.play(
            Create(arrow_top),
            Create(arrow_bottom)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Central Title for the theorem
        ftc_title = Text("The Fundamental Theorem\nof Calculus", font_size=28, color="#FFD700", line_spacing=1)
        # Position theorem title using refined area and scale from Issues 51, 57
        self.place_in_area(ftc_title, "E2", "F5", scale_factor=0.9)

        self.play(GrowFromCenter(ftc_title))
        self.wait(3)

        # Fade out highlight for final look
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
