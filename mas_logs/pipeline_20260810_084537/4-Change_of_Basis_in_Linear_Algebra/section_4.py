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
        self.setup_layout("Application: Why Do We Care?", [
            "Choose bases to simplify complex mathematical calculations.",
            "Diagonalization makes shapes appear as simple rectangles.",
            "Robots use local bases for easier control."
        ])
        
        # Load asset
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # === Animation for Lecture Line 1 ===
        # Use SVG asset as the "complex system"
        robot_complex = SVGMobject(robot_asset, color=BLUE)
        self.place_in_area(robot_complex, 'A1', 'B3', scale_factor=0.7)
        self.play(FadeIn(robot_complex))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Diagonalization -> rectangle transformation
        rect = Rectangle(width=2, height=1, color=GREEN)
        self.place_in_area(rect, 'C1', 'D3', scale_factor=0.7)
        self.play(Transform(robot_complex, rect))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Highlight and local basis representation
        # Re-use the existing robot object or create a simplified local-basis view
        robot_local = SVGMobject(robot_asset, color=YELLOW)
        self.place_in_area(robot_local, 'E1', 'F3', scale_factor=0.7)
        self.play(Create(robot_local))
        self.lecture[2].set_color("#FFFF00")
