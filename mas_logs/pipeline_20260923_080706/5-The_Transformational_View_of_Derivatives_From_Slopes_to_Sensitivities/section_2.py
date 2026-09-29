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
            "View derivatives as a transformation machine.",
            "It maps curve points to slope values.",
            "Red curve input, blue derivative output.",
            "Each x-value has a unique slope.",
            "The derivative is now a function."
        ]
        self.setup_layout("The Transformational Shift: Derivative as a Function", lecture_lines)
        
        # Asset loader (simplified as placeholders for this environment)
        machine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        
        # Create Axes for Right side
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 4, 1], axis_config={"include_tip": True}).scale(0.5)
        f = axes.plot(lambda x: x**2, color="#FF0000")
        f_prime = axes.plot(lambda x: 2*x, color="#00FF00")
        
        curve_group = VGroup(axes, f, f_prime)
        self.place_in_area(curve_group, 'B3', 'F6', scale_factor=0.65)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(machine_icon, 'A2', scale_factor=0.5)
        self.play(FadeIn(machine_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.play(Create(f))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.play(Create(f_prime))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        dot = Dot(color="#FFFFFF")
        # Dot updater to trace f(x) and show correspond to f'(x)
        dot.add_updater(lambda d: d.move_to(axes.c2p(axes.p2c(d.get_center())[0], (axes.p2c(d.get_center())[0])**2)))
        self.play(FadeIn(dot))
        self.play(MoveAlongPath(dot, f), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        zero_point = Dot(axes.c2p(0, 0), color="#00FFFF")
        machine_zero = machine_icon.copy().scale(0.5)
        self.place_at_grid(machine_zero, 'D2')
        self.play(FadeIn(zero_point), FadeIn(machine_zero))
        self.wait(2)
