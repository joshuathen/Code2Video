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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Curves", [
            "The SIR graph shows disease dynamics.",
            "Infection peaks as Susceptible count drops.",
            "Flattening the curve reduces peak infection.",
            "Social distancing lowers the peak significantly.",
            "Geometry explains real-world policy impact."
        ])
        
        # Colors for lecture lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Assets
        person = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
        hospital = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg")
        
        # Setup graph elements
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False})
        
        s_curve = axes.plot(lambda x: 0.9 * np.exp(-0.2 * x), color=BLUE)
        i_curve = axes.plot(lambda x: 0.5 * x * np.exp(-0.4 * x), color=RED)
        r_curve = axes.plot(lambda x: 0.9 * (1 - np.exp(-0.2 * x)), color=GREEN)
        curves = VGroup(s_curve, i_curve, r_curve)

        # Apply layout fixes based on critiques
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.65)
        self.place_in_area(curves, 'B3', 'E5', scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]), Create(axes), FadeIn(person))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]), Create(s_curve), Create(i_curve), person.animate.shift(RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        new_i_curve = axes.plot(lambda x: 0.25 * x * np.exp(-0.4 * x), color=RED)
        self.play(self.lecture[2].animate.set_color(colors[2]), Transform(i_curve, new_i_curve), FadeIn(hospital))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]), i_curve.animate.set_stroke(width=6), hospital.animate.scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]), Create(r_curve))
        self.wait(2)
