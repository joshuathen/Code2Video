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
        self.setup_layout("The Peano Curve Construction", [
            "Start with a single simple line segment.",
            "Split it into smaller sub-segments recursively.",
            "Folding eventually fills the entire 2D area."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Start with a single simple line segment.
        self.lecture[0].set_color("#FFFFFF")
        # Use SVG asset as required by instruction
        grid_square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg", color=WHITE)
        # Apply Critic Fix: self.place_in_area(grid_square, 'B2', 'D4', scale_factor=0.6)
        self.place_in_area(grid_square, 'B2', 'D4', scale_factor=0.6)
        
        # Apply Critic Fix: self.place_at_grid(seed_label, 'B2', scale_factor=0.7)
        seed_label = Text("Seed", font_size=18).next_to(grid_square, UP)
        self.place_at_grid(seed_label, 'B2', scale_factor=0.7)
        
        self.play(Create(grid_square), Write(seed_label))

        # === Animation for Lecture Line 2 ===
        # Split it into smaller sub-segments recursively.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#CCCCCC")
        
        grid_3x3 = VGroup(*[
            Square(side_length=0.8, color="#CCCCCC") 
            for _ in range(9)
        ]).arrange_in_grid(rows=3, cols=3, buff=0)
        
        # Apply Critic Fix: self.place_in_area(grid_3x3, 'B3', 'E6', scale_factor=0.7)
        self.place_in_area(grid_3x3, 'B3', 'E6', scale_factor=0.7)
        
        self.play(ReplacementTransform(grid_square, grid_3x3))
        
        # Connect centers
        path = VMobject(color="#FF00FF")
        centers = [sq.get_center() for sq in grid_3x3]
        path.set_points_smoothly([centers[0], centers[1], centers[2], centers[5], centers[4], centers[3], centers[6], centers[7], centers[8]])
        self.play(Create(path))

        # === Animation for Lecture Line 3 ===
        # Folding eventually fills the entire 2D area.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # Higher density curve
        dense_path = VMobject(color="#00FFFF")
        # Simplified recursive fold
        dense_path.set_points_smoothly([
            UP*1.2 + LEFT*1.2, UP*1.2 + LEFT*0.4, UP*1.2 + RIGHT*0.4, 
            UP*0.4 + RIGHT*0.4, UP*0.4 + LEFT*0.4, DOWN*0.4 + LEFT*0.4,
            DOWN*0.4 + RIGHT*0.4, DOWN*1.2 + RIGHT*0.4, DOWN*1.2 + LEFT*1.2
        ])
        dense_path.shift(grid_3x3.get_center())
        
        self.play(ReplacementTransform(path, dense_path), FadeOut(grid_3x3), FadeOut(seed_label))
        self.wait(1)
