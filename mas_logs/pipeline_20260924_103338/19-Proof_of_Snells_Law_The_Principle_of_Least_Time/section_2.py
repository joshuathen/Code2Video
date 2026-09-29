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
        self.setup_layout("Fermat’s Principle & Prerequisites", [
            "Light follows the path of minimum time.",
            "We define indices of refraction for media.",
            "A ray crosses the interface at point P."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fermat's Principle formula: 'Time = Distance / Speed' (#00FF00)
        # Added asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/interface.svg
        interface_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/interface.svg", color=WHITE)
        time_formula = MathTex(r"T = \frac{D}{S}", color="#00FF00")
        
        formula_group = VGroup(time_formula, interface_icon).arrange(DOWN)
        self.place_at_grid(formula_group, 'B3', scale_factor=1.0) # Updated per issue 29/44
        self.play(Write(time_formula), FadeIn(interface_icon))
        self.lecture[0].set_color("#00FF00")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Label variables as 'D' for distance and 'S' for speed. (#00FFFF)
        d_label = Text("D = Distance", color="#00FFFF", font_size=24)
        s_label = Text("S = Speed", color="#00FFFF", font_size=24)
        v_group = VGroup(d_label, s_label).arrange(DOWN)
        self.place_at_grid(v_group, 'D4', scale_factor=0.8) # Updated per issue 30/45
        self.play(FadeIn(v_group))
        self.lecture[1].set_color("#00FFFF")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Show a slider to adjust speed; label as 'Speed Slider'. (#FF00FF)
        # Added asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg
        slider_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg", color=WHITE)
        slider_label = Text("Speed Slider", color="#FF00FF", font_size=20)
        
        slider_group = VGroup(slider_label, slider_icon).arrange(DOWN)
        self.place_at_grid(slider_group, 'B5', scale_factor=0.8) # Updated per issue 31/46
        
        self.play(FadeIn(slider_icon), Write(slider_label))
        self.lecture[2].set_color("#FF00FF")
        
        # Animating the icon movement (simulating slider motion)
        self.play(slider_icon.animate.shift(LEFT*0.3), run_time=1.0)
        self.play(slider_icon.animate.shift(RIGHT*0.6), run_time=1.0)
        self.wait(2)
