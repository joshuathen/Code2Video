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
        title = "Prerequisite Knowledge: The Basis Vectors"
        lines = [
            "Basis vectors i-hat and j-hat define the unit square.", 
            "The matrix tells us where these vectors land.", 
            "Their new positions determine the transformation's resulting area."
        ]
        self.setup_layout(title, lines)

        # Anchors and points - Shifted origin to D3 to accommodate rightward movement
        origin = self.grid["D3"]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Basis vectors: i-hat ends at D4 (1,0), j-hat ends at C3 (0,1)
        i_hat = Arrow(origin, self.grid["D4"], buff=0, color="#FF0000", stroke_width=4)
        j_hat = Arrow(origin, self.grid["C3"], buff=0, color="#00FF00", stroke_width=4)
        
        # Labels - Adjusted per Issues 35 & 37
        i_label = Text("i-hat", font_size=16, color="#FF0000")
        self.place_at_grid(i_label, "E4")
        
        j_label = Text("j-hat", font_size=16, color="#00FF00")
        self.place_at_grid(j_label, "C2")
        
        # Unit square - Adjusted per Issue 36
        unit_square = Rectangle(width=1, height=1, color="#FFFFFF")
        unit_square.set_style(stroke_width=2)
        unit_square = DashedVMobject(unit_square)
        self.place_in_area(unit_square, "C3", "D4")
        
        self.play(
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            FadeIn(i_label),
            FadeIn(j_label),
            Create(unit_square),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Matrix components target positions
        # i-hat moves to (3,0) -> D6
        # j-hat moves to (0,2) -> B3
        target_i = Arrow(origin, self.grid["D6"], buff=0, color="#FF0000", stroke_width=4)
        target_j = Arrow(origin, self.grid["B3"], buff=0, color="#00FF00", stroke_width=4)
        
        # Labels target positions - Adjusted per Issues 35 & 37
        target_i_label = Text("i-hat", font_size=16, color="#FF0000")
        self.place_at_grid(target_i_label, "E6")
        
        target_j_label = Text("j-hat", font_size=16, color="#00FF00")
        self.place_at_grid(target_j_label, "B2")
        
        # Transformed rectangle - Adjusted per Issue 36
        # Rectangle starts at origin D3, width 3, height 2. Top-left B3, bottom-right D6
        transformed_rect = Rectangle(width=3, height=2, color="#FFFFFF")
        transformed_rect.set_style(stroke_width=3, fill_opacity=0.2)
        self.place_in_area(transformed_rect, "B3", "D6")

        self.play(
            Transform(i_hat, target_i),
            Transform(j_hat, target_j),
            Transform(i_label, target_i_label),
            Transform(j_label, target_j_label),
            ReplacementTransform(unit_square, transformed_rect),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Highlighting the final area - Adjusted per Issue 37
        area_text = Text("Area = 6", font_size=20, color=WHITE)
        self.place_at_grid(area_text, "C5") # Center of the transformed area (between B3 and D6 columns 3-6)
        
        self.play(
            transformed_rect.animate.set_stroke(color=YELLOW, width=5),
            Write(area_text),
            run_time=1.5
        )
        self.wait(2)
        
        # Final cleanup highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
