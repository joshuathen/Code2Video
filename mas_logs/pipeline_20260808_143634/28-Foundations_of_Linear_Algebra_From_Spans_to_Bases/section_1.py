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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors are arrows from origin to points.",
            "Combine vectors by scaling and adding.",
            "This creates new vectors in space.",
            "The robot arm shows reachability.",
            "Linear combinations define movement space."
        ]
        self.setup_layout("Prerequisites & Linear Combinations", lecture_lines)
        
        # Load assets
        robot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Create vectors
        v1 = Arrow(ORIGIN, RIGHT + UP, color=BLUE)
        v2 = Arrow(ORIGIN, RIGHT * 0.5 + DOWN * 1.5, color=GREEN)
        vector_group = VGroup(v1, v2)
        
        # Positioning
        self.place_at_grid(v1, 'B2', scale_factor=0.6)
        self.place_at_grid(v2, 'E5', scale_factor=0.6)
        self.place_in_area(vector_group, 'C4', 'F6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(v1), FadeIn(v2), FadeIn(robot_svg))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Scalar multiplication
        v1_scaled = Arrow(ORIGIN, (RIGHT + UP) * 1.5, color=BLUE)
        self.play(Transform(v1, v1_scaled))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        v_sum = Vector(v1.get_end() + v2.get_end(), color=YELLOW)
        self.play(Create(v_sum))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Visualizing robot arm movement
        self.place_at_grid(robot_svg, 'D4', scale_factor=0.5)
        self.play(Indicate(robot_svg))
        self.lecture[3].set_color(RED)

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(robot_svg), FadeOut(v1), FadeOut(v2), FadeOut(v_sum))
        self.lecture[4].set_color(ORANGE)
        self.wait(1)
