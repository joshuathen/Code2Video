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
        lecture_lines = ["Base-3 uses digits zero, one, and two.", "Three-way branching needs ternary logic.", "Robot movement maps to ternary sequences."]
        self.setup_layout("Prerequisite: The Logic of Base-3", lecture_lines)
        
        # Load assets
        robot_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # Create elements
        node_a = SVGMobject(robot_svg, color=WHITE)
        node_b = SVGMobject(robot_svg, color=WHITE)
        node_c = SVGMobject(robot_svg, color=WHITE)
        
        # Place as per Issue 21/22/23
        self.place_at_grid(node_a, 'C3', scale_factor=0.3)
        self.place_at_grid(node_b, 'E2', scale_factor=0.3)
        self.place_at_grid(node_c, 'E4', scale_factor=0.3)
        
        label_a = Text("A", font_size=24).next_to(node_a, UP, buff=0.2)
        label_b = Text("B", font_size=24).next_to(node_b, DOWN, buff=0.2)
        label_c = Text("C", font_size=24).next_to(node_c, DOWN, buff=0.2)

        path_ab = Line(node_a.get_center(), node_b.get_center(), color=WHITE)
        path_bc = Line(node_b.get_center(), node_c.get_center(), color=WHITE)
        path_ac = Line(node_a.get_center(), node_c.get_center(), color=WHITE)
        
        robot_end = SVGMobject(robot_svg, color=RED).scale(0.2).move_to(node_c.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"), 
                  FadeIn(node_a), FadeIn(node_b), FadeIn(node_c), 
                  Write(label_a), Write(label_b), Write(label_c))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), 
                  Create(path_ab.set_color("#FFFF00")), 
                  Create(path_bc.set_color("#FFFFFF")), 
                  Create(path_ac.set_color("#FF0000")))
        
        label_0 = Text("0", font_size=20, color="#FFFF00").move_to(path_ab.get_center() + UP*0.2)
        label_1 = Text("1", font_size=20, color="#FFFFFF").move_to(path_bc.get_center() + DOWN*0.2)
        label_2 = Text("2", font_size=20, color="#FF0000").move_to(path_ac.get_center() + UP*0.2)
        
        self.play(Write(label_0), Write(label_1), Write(label_2), FadeIn(robot_end))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
