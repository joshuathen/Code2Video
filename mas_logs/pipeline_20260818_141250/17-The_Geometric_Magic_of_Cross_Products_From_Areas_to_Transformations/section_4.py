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
        self.setup_layout("Application: The Physical World (Torque)", [
            "Torque is defined as r cross F.",
            "Cross product captures leverage effects.",
            "Perpendicular forces cause rotation."
        ])
        
        # Asset paths
        wrench_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg"
        bolt_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bolt.svg"
        
        # Load Assets
        wrench = SVGMobject(wrench_path)
        bolt_torque = SVGMobject(bolt_path)
        bolt_rotation = SVGMobject(bolt_path)
        
        # === Animation for Lecture Line 1 ===
        # Using SVG Assets instead of basic arrows
        self.place_at_grid(wrench, 'C2', scale_factor=0.8)
        wrench.set_color(WHITE)
        
        # Fixed LaTeX string by reducing backslashes
        torque_label = MathTex(r"\vec{\tau} = \vec{r} \times \vec{F}", color=WHITE)
        self.place_in_area(torque_label, 'A3', 'B4', scale_factor=0.9)
        
        self.play(FadeIn(wrench), Write(torque_label))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(bolt_torque, 'C4', scale_factor=0.8)
        bolt_torque.set_color("#FF4500")
        
        leverage_text = Text("Leverage = |r||F|sin(θ)", font_size=20, color="#FF4500")
        self.place_in_area(leverage_text, 'E2', 'F5', scale_factor=0.7)
        
        self.play(FadeIn(bolt_torque), Write(leverage_text))
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(bolt_rotation, 'C3', scale_factor=0.8)
        bolt_rotation.set_color("#00CED1")
        
        rotation_arrow = Arc(radius=0.5, start_angle=0, angle=PI/2, color="#00CED1")
        rotation_arrow.move_to(self.grid['C3'] + RIGHT*0.5 + UP*0.5)
        
        self.play(FadeIn(bolt_rotation), Create(rotation_arrow))
        self.play(self.lecture[2].animate.set_color("#00CED1"))
        self.wait(2)
