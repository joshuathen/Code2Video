from manim import *
import os

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
            "Let's apply this to a Robot Scout.",
            "Number of trials n is 5.",
            "Successes k we need is 4.",
            "Probability p is 0.7.",
            "Calculate the total probability for this outcome."
        ]
        self.setup_layout("Application: The Robot Scout", lecture_lines)
        
        # Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        scout_label = Text("Scout", color="#1E90FF", font_size=24)
        scout_group = VGroup(robot, scout_label).arrange(DOWN)
        
        # Grid visualizers
        path_nodes = VGroup(*[Circle(radius=0.2, color=WHITE) for _ in range(5)]).arrange(RIGHT, buff=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#1E90FF")
        self.place_at_grid(scout_group, 'C4', scale_factor=0.6)
        self.play(FadeIn(scout_group))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_in_area(path_nodes, 'D2', 'D6', scale_factor=0.7)
        self.play(Create(path_nodes))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.play(path_nodes[0:4].animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        p_val = Text("p = 0.7", color="#FF4500", font_size=20)
        self.place_at_grid(p_val, 'B4', scale_factor=0.8)
        self.play(Write(p_val))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#DA70D6")
        res_label = Text("P(X=4) = 0.3601", color="#FFD700", font_size=24)
        self.place_at_grid(res_label, 'E4', scale_factor=0.8)
        
        self.play(robot.animate.move_to(path_nodes[3].get_center()), run_time=2)
        self.play(Write(res_label))
        self.wait(1)
