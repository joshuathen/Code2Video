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
        lecture_lines = [
            "Fixed points satisfy the equation f(z) equals z.",
            "Derivatives determine the stability of these points.",
            "Attractors pull nearby points into their orbit."
        ]
        self.setup_layout("Fixed Points and Stability", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Fixed points satisfy the equation f(z) = z.
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg
        magnet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg")
        self.place_at_grid(magnet, 'B5', scale_factor=0.8)
        
        fixed_pt = Circle(radius=0.2, color="#1E90FF", fill_opacity=0.5)
        fixed_pt.move_to(magnet.get_center())
        
        label_f = MathTex("f(z)=z").scale(0.8)
        # Use place_in_area to position label clearly
        self.place_in_area(label_f, 'A4', 'A6', scale_factor=0.8)
        
        self.play(FadeIn(magnet), FadeIn(fixed_pt), Write(label_f))
        self.lecture[0].set_color("#1E90FF")

        # === Animation for Lecture Line 2 ===
        # Derivatives determine the stability of these points.
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg
        vortex = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg")
        self.place_at_grid(vortex, 'E5', scale_factor=0.8)
        
        unstable_pt = Cross(color="#FF0000", scale_factor=0.3)
        unstable_pt.move_to(vortex.get_center())
        
        label_u = Text("Unstable", font_size=18, color="#FF0000").next_to(unstable_pt, DOWN, buff=0.1)
        
        self.play(FadeIn(vortex), FadeIn(unstable_pt), Write(label_u))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Attractors pull nearby points into their orbit.
        orbit_dot = Dot(color="#00FF00")
        start_pos = self.grid['B2'] # Start on the left to show movement
        orbit_dot.move_to(start_pos)
        
        self.add(orbit_dot)
        self.play(
            orbit_dot.animate.move_to(fixed_pt.get_center()),
            run_time=2,
            rate_func=linear
        )
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
