from manim import *
import numpy as np

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
        # 1. Setup Layout
        title_text = "Normalizing the Space (The Posterior)"
        lecture_lines = [
            "Let's combine the remaining areas to form a new space.",
            "This total area represents all possible glinting events.",
            "We normalize this space back into a unit square.",
            "The gold region's relative size gives the new probability.",
            "This updated value is called the posterior probability."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        GOLD_CLR = "#FFD700"
        GREY_CLR = "#888888"

        # Create components
        # Proportional sizes: Gold is 0.08, Other is 0.18.
        # We start with them at a visible scale before normalizing.
        gold_rect = Rectangle(width=0.8, height=2.0, fill_color=GOLD_CLR, fill_opacity=0.8, stroke_width=2)
        grey_rect = Rectangle(width=1.8, height=2.0, fill_color=GREY_CLR, fill_opacity=0.5, stroke_width=2)
        grey_rect.next_to(gold_rect, RIGHT, buff=0)
        diagram_group = VGroup(gold_rect, grey_rect)
        
        label_h = Text("Gold", color=GOLD_CLR, font_size=28)
        label_not_h = Text("Other", color=GREY_CLR, font_size=28)
        label_e = Text("Evidence: Glint", color=YELLOW, font_size=24)
        
        posterior_formula = Text("0.08 / 0.26 ≈ 31%", font_size=32, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Line 1: self.place_in_area(diagram_group, 'B2', 'E5', scale_factor=1.0)
        # Line 2: self.place_at_grid(label_h, 'A2', scale_factor=0.8)
        # Line 3: self.place_at_grid(label_not_h, 'A4', scale_factor=0.8)
        self.play(self.lecture[0].animate.set_color(GOLD_CLR))
        
        self.place_in_area(diagram_group, 'B2', 'E5', scale_factor=1.0)
        self.place_at_grid(label_h, 'A2', scale_factor=0.8)
        self.place_at_grid(label_not_h, 'A4', scale_factor=0.8)
        
        self.play(
            FadeIn(diagram_group),
            Write(label_h),
            Write(label_not_h)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 4: self.place_at_grid(label_e, 'C1', scale_factor=0.8)
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        self.place_at_grid(label_e, 'C1', scale_factor=0.8)
        
        brace = Brace(diagram_group, DOWN, color=WHITE)
        brace_text = brace.get_text("Total Evidence: 0.26").scale(0.8)
        
        self.play(
            Create(brace),
            Write(brace_text),
            Write(label_e)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Scale the grouped rectangles to occupy a 4x4 area (filling the original square's space)
        # We'll use a unit square frame for reference.
        unit_square = Rectangle(width=4.0, height=4.0, color=WHITE, stroke_dash_array=[5, 5])
        self.place_in_area(unit_square, 'B2', 'E5')
        
        # Calculate target position for labels to stay above the scaled box
        self.play(
            diagram_group.animate.stretch_to_fit_width(4.0).stretch_to_fit_height(4.0).move_to(unit_square.get_center()),
            brace.animate.stretch_to_fit_width(4.0).next_to(unit_square, DOWN, buff=0.1),
            brace_text.animate.next_to(unit_square, DOWN, buff=0.5),
            label_h.animate.next_to(unit_square, UP, buff=0.2).set_x(unit_square.get_left()[0] + 0.6),
            label_not_h.animate.next_to(unit_square, UP, buff=0.2).set_x(unit_square.get_right()[0] - 1.2),
            FadeIn(unit_square)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GOLD_CLR))
        
        # Math text centered in the gold section
        # gold_rect width is now 4.0 * (0.08 / 0.26) = ~1.23
        math_in_box = Text("0.08 / 0.26 ≈ 31%", font_size=18, color=WHITE).move_to(gold_rect.get_center())
        
        self.play(Write(math_in_box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: self.place_in_area(posterior_formula, 'F2', 'F5', scale_factor=0.75)
        self.play(self.lecture[4].animate.set_color(GOLD_CLR))
        
        self.place_in_area(posterior_formula, 'F2', 'F5', scale_factor=0.75)
        
        posterior_label = Text("Posterior P(Gold|Glint)", color=GOLD_CLR, font_size=20)
        posterior_label.next_to(gold_rect, UP, buff=-0.5)
        
        self.play(
            Write(posterior_formula),
            FadeOut(label_h),
            Write(posterior_label)
        )
        self.wait(2)
