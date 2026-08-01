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
        self.setup_layout(
            "Prerequisite Knowledge: The Language of Math", 
            [
                "Effective explanations start with a clear toolbox.", 
                "We must define terms like radius and pi.", 
                "Establish baseline tools before solving the problem."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create toolbox outline
        box_outline = Rectangle(width=4, height=3, color="#DEB887")
        self.place_in_area(box_outline, 'B2', 'E5')
        
        # Label
        box_label = Text("Explanation Toolbox", font_size=20, color="#FFFFFF")
        self.place_at_grid(box_label, 'B3', scale_factor=0.8)
        box_label.shift(UP * 0.5) # Slight manual adjustment within grid constraints to stay above box center
        
        self.play(Create(box_outline), Write(box_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight current line, revert previous
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define tools - Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        radius_tool = Text("r", slant=ITALIC, color="#00FFFF")
        pi_tool = Text("π", color="#FFD700")
        square_tool = Text("x²", color="#ADFF2F")
        
        # Place tools initially outside/at edges
        self.place_at_grid(radius_tool, 'A1', scale_factor=1.5)
        self.place_at_grid(pi_tool, 'A6', scale_factor=1.5)
        self.place_at_grid(square_tool, 'F1', scale_factor=1.5)
        
        self.play(FadeIn(radius_tool), FadeIn(pi_tool), FadeIn(square_tool))
        
        # Target positions inside the box area
        target_r = self.grid['D2']
        target_p = self.grid['D3']
        target_s = self.grid['D4']
        
        self.play(
            radius_tool.animate.move_to(target_r),
            pi_tool.animate.move_to(target_p),
            square_tool.animate.move_to(target_s),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight current line, revert previous
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Box lid: represent with a line at the top of the rectangle
        lid = Line(
            start=box_outline.get_corner(UL), 
            end=box_outline.get_corner(UR), 
            color="#DEB887", 
            stroke_width=6
        )
        # Position lid slightly above to animate "closing"
        lid.shift(UP * 0.5)
        self.play(FadeIn(lid))
        self.play(lid.animate.move_to(box_outline.get_top()), run_time=0.8)
        
        # Glow effect
        glow = SurroundingRectangle(box_outline, color=WHITE, buff=0.1, stroke_width=2)
        self.play(Create(glow), run_time=0.5)
        self.play(Indicate(box_outline, color=WHITE), FadeOut(glow), run_time=1)
        
        self.wait(2)
        
        # Final cleanup highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
