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
        lecture_lines = [
            "Robots move with steps of size 1, 2, 3.",
            "We want total distance N, step count mod 3.",
            "Evaluate at cube roots to isolate paths.",
            "The generating function encodes all valid robot movements.",
            "Complex evaluation filters the paths we desire."
        ]
        self.setup_layout("Application: Counting Combinations for Robotic Movements", lecture_lines)
        
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # === Animation for Lecture Line 1 ===
        robot1 = SVGMobject(robot_asset)
        self.place_at_grid(robot1, 'B2', scale_factor=0.5)
        robot1.set_color("#FFD700")
        self.play(FadeIn(robot1))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        sum_mod = MathTex(r"S \equiv 0 \pmod 3").set_color("#FFFFFF")
        self.place_at_grid(sum_mod, 'B4', scale_factor=0.8)
        self.play(Write(sum_mod))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        circle = Circle(radius=1.0, color="#FF6347")
        roots = VGroup(*[Dot(point=circle.point_from_proportion(i/3), color="#FF6347") for i in range(3)])
        group3 = VGroup(circle, roots)
        self.place_at_grid(group3, 'E3', scale_factor=0.8)
        self.play(Create(group3))
        self.lecture[2].set_color("#FF6347")

        # === Animation for Lecture Line 4 ===
        gf = MathTex(r"f(x) = x + x^2 + x^3").set_color("#00FFFF")
        self.place_at_grid(gf, 'E5', scale_factor=0.8)
        self.play(Write(gf))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        robot2 = SVGMobject(robot_asset)
        self.place_at_grid(robot2, 'C5', scale_factor=0.5)
        robot2.set_color("#00FF00")
        self.play(FadeIn(robot2))
        self.lecture[4].set_color("#00FF00")
        self.wait(1)
