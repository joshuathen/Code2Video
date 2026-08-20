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
            "Cheetah motion models position over time.",
            "The derivative transforms position into velocity.",
            "Then, it transforms velocity into acceleration.",
            "Each step tracks increasing motion intensity.",
            "Derivatives describe changing speed accurately."
        ]
        self.setup_layout("Application: The Cheetah’s Acceleration", lecture_lines)
        
        # Setup visualization elements
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], axis_config={"include_tip": False})
        # Applying layout fix as per issue 35 & 37: place in B3-F6, scale 0.6
        graph_pos = self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        
        # Function: x(t) = 0.2t^2
        position_curve = axes.plot(lambda t: 0.2 * t**2, color=BLUE)
        
        # Cheetah SVG
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah, 'A2', scale_factor=0.3)
        
        # Velocity marker
        dot = Dot(color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.add(graph_pos, position_curve, cheetah)
        self.play(Create(dot), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        # Add tangent line
        tangent = Line(color=YELLOW)
        self.play(Create(tangent))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        accel_vec = Arrow(start=ORIGIN, end=UP*0.5, color="#FFD700")
        # Applying layout fix as per issue 36: place at D4, scale 0.7
        self.place_at_grid(accel_vec, 'D4', scale_factor=0.7)
        self.play(GrowArrow(accel_vec))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)
