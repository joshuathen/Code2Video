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
        # Setup layout
        title_text = "The Prerequisite: The Chain Rule Reminder"
        lecture_lines = [
            "Remember the Chain Rule: it's our engine for differentiation.",
            "Treat y as a hidden function nested inside x.",
            "Every y-derivative must leave a dy/dx tail behind."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Description: Show the expression (y)^3 in white (#FFFFFF) inside a light green (#90EE90) container.
        
        expr_y3 = Text("(y)^3", color=WHITE)
        container = RoundedRectangle(
            corner_radius=0.2, 
            color="#90EE90", 
            fill_opacity=0.1,
            stroke_width=4
        )
        container.surround(expr_y3, buff=0.5)
        
        # Create a group for positioning
        initial_group = VGroup(container, expr_y3)
        # Fix for Issue 17: Use area B2 to C5 and scale 1.0
        self.place_in_area(initial_group, 'B2', 'C5', scale_factor=1.0)
        
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            Create(container),
            Write(expr_y3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Treat y as a hidden function nested inside x.
        # Action: Color line 2 and pulse the container to emphasize the "nesting".
        
        self.play(self.lecture[1].animate.set_color("#90EE90"))
        self.play(
            container.animate.set_stroke(width=10),
            run_time=0.4
        )
        self.play(
            container.animate.set_stroke(width=4),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Transform (y)^3 into 3y^2 in white (#FFFFFF) while the container fades.
        # Attach the term (dy/dx) in orange (#FFA500) next to 3y^2 as the 'chain rule tail'.
        
        final_y2 = Text("3y^2", color=WHITE)
        tail = Text("dy/dx", color="#FFA500")
        final_group = VGroup(final_y2, tail).arrange(RIGHT, buff=0.3)
        
        # Fix for Issue 18 & 19: Position final group in area D2 to E5 and scale 1.0
        self.place_in_area(final_group, 'D2', 'E5', scale_factor=1.0)
        
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        
        # Transformation
        self.play(
            ReplacementTransform(expr_y3, final_y2),
            FadeOut(container)
        )
        # Adding the "tail"
        self.play(
            Write(tail)
        )
        self.wait(3)
