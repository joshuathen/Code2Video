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
        self.setup_layout(
            "The Wave Equation: Oscillations in Space-Time",
            ["The Wave Equation describes oscillatory motion over time.",
             "It uses second-order time derivatives: u_tt = c²u_xx.",
             "Disturbances propagate as waves through a medium."]
        )

        # Visualization elements
        axes = Axes(x_range=[-3, 3], y_range=[-1.5, 1.5], axis_config={"include_numbers": False}).scale(0.5)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg
        string_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_in_area(axes, 'B4', 'D6', scale_factor=0.75)
        self.place_at_grid(string_icon, 'B1', scale_factor=0.5)
        
        # Use ValueTracker instead of always_redraw to stay within constraints
        time_tracker = ValueTracker(0)
        def update_wave(m):
            t = time_tracker.get_value()
            new_wave = axes.plot(lambda x: 0.8 * np.exp(-(x - 2*np.sin(t))**2), x_range=[-3, 3], color="#FFD700")
            m.become(new_wave)
        
        wave = axes.plot(lambda x: 0, x_range=[-3, 3], color="#FFD700")
        wave.add_updater(update_wave)
        self.add(wave)
        
        self.play(time_tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        equation = MathTex(r"u_{tt} = c^2 u_{xx}", color="#00FFFF")
        self.place_at_grid(equation, 'D3', scale_factor=0.9)
        self.play(Write(equation))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF69B4")
        self.wait(3)
