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

class Section3Scene(TeachingScene):
    def construct(self):
        # 1. Setup the layout with title and lecture lines
        self.setup_layout(
            "The Golden Rule: Treat Y as y(x)", 
            [
                "Treat y as a hidden function of x.",
                "When deriving y, we must use the Chain Rule.",
                "Think of y squared as a package containing x.",
                "The derivative is 2y times the tail, dy/dx.",
                "Every y derivative leaves a dy/dx link behind."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Show 'y' morphing into 'y(x)' in cyan #00FFFF.
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        y_text = Text("y", color="#00FFFF")
        self.place_at_grid(y_text, "B3", scale_factor=1.5)
        self.play(Write(y_text))
        self.wait(0.5)
        
        yx_text = Text("y(x)", color="#00FFFF")
        self.place_at_grid(yx_text, "B3", scale_factor=1.5)
        self.play(ReplacementTransform(y_text, yx_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "When deriving y, we must use the Chain Rule."
        # No specific visual step required here other than highlighting text.
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Write 'd/dx [ y^2 ]' in white #FFFFFF.
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        d_dx_expr = Text("d/dx [ y^2 ]", color="#FFFFFF")
        self.place_at_grid(d_dx_expr, "C3", scale_factor=1.2)
        self.play(Write(d_dx_expr))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transform to '2y' followed by an empty 'Chain Link' slot.
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        res_2y = Text("2y", color="#FFFFFF")
        # Empty chain link slot represented by a box
        slot_box = Square(side_length=0.7, color="#FFFFFF", stroke_width=2)
        result_group = VGroup(res_2y, slot_box).arrange(RIGHT, buff=0.3)
        self.place_at_grid(result_group, "D3", scale_factor=1.2)
        
        self.play(TransformFromCopy(d_dx_expr, result_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fill the slot with 'dy/dx' in bold gold #FFD700.
        # Highlight the 'dy/dx' tail with a glowing border.
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        # 'dy/dx' in gold (bold effect simulated by stroke)
        dydx_tail = Text("dy/dx", color="#FFD700")
        dydx_tail.set_stroke(width=1)
        dydx_tail.move_to(slot_box.get_center())
        
        self.play(FadeIn(dydx_tail, scale=0.5))
        self.wait(0.5)
        
        # Add glowing border
        glow_rect = SurroundingRectangle(dydx_tail, color="#FFD700", buff=0.1)
        glow_rect.set_fill("#FFD700", opacity=0.3)
        self.play(Create(glow_rect))
        self.wait(2)
