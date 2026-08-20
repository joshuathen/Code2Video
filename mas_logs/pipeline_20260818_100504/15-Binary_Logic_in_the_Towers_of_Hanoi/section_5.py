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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Binary logic models recursive algorithms.", "It navigates complex state spaces efficiently.", "Robotic arms use binary clocks precisely."]
        self.setup_layout("Real-World Application", lecture_lines)
        
        # Elements
        # Animation 1 Elements: Function stack
        stack_frame = Rectangle(width=3, height=2, color=WHITE)
        stack_label = Text("stack", color="#FFFF00", font_size=20)
        stack_group = VGroup(stack_frame, stack_label).arrange(UP)
        self.place_in_area(stack_group, 'A1', 'B6', scale_factor=0.8)

        # Animation 2 Elements: Hanoi tree path (simplified as nodes)
        nodes = VGroup(*[Dot(color="#00FFFF") for _ in range(5)]).arrange(RIGHT, buff=0.5)
        path = VGroup(*[Line(nodes[i].get_center(), nodes[i+1].get_center(), color="#00FFFF") for i in range(4)])
        tree_group = VGroup(nodes, path)
        self.place_in_area(tree_group, 'C1', 'D6', scale_factor=0.8)

        # Animation 3 Elements: Robot arm icon
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot_icon, 'F3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(stack_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(Create(tree_group))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(robot_icon))
        self.lecture[2].set_color("#FFFFFF")
        
        # Clock signal effect
        clock_signal = Dot(color="#FFFFFF").move_to(robot_icon.get_right())
        self.play(Indicate(clock_signal))
        
        self.wait(2)
