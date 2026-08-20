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
        lecture_lines = [
            "A vector space obeys eight formal rules.",
            "Closure keeps vectors inside the space.",
            "Addition and multiplication are consistently defined.",
            "These axioms define our abstract playground.",
            "Like a robot's reliable instruction set."
        ]
        self.setup_layout("The Eight Axioms: Defining the Playground", lecture_lines)
        self.lecture.set_opacity(0)
        
        # Setup Animation Elements
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        robot1 = SVGMobject(robot_asset).scale(0.5)
        robot2 = SVGMobject(robot_asset).scale(0.3)
        
        dots = VGroup(*[Dot(color="#00FFFF") for _ in range(8)])
        circle = VGroup()
        for i, dot in enumerate(dots):
            angle = i * PI / 4
            # Placed visually relative to the central area
            dot.move_to(np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle), 0]))
            circle.add(dot)
            
        axiom_text = Text("Axiom Space", color="#FFFFFF", font_size=20)
        
        # Combined group for the playground
        playground = VGroup(circle, robot1, axiom_text)
        self.place_in_area(playground, 'C3', 'E5', scale_factor=0.9)
        
        structure_label = Text("Structure", color="#FFFFFF", font_size=20)
        self.place_at_grid(structure_label, 'B4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(playground))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.play(
            *[dot.animate.set_color("#FFFF00") for dot in dots],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        lines = VGroup()
        for i in range(8):
            lines.add(Line(dots[i].get_center(), dots[(i+1)%8].get_center(), color="#FF00FF"))
        self.play(Create(lines))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        self.play(Write(axiom_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        self.play(FadeIn(robot2.next_to(structure_label, RIGHT, buff=0.1)))
        self.play(Rotate(structure_label, angle=0.2, about_point=structure_label.get_center()))
        self.wait(2)
