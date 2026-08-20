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
        self.setup_layout("The Problem of 'Instantaneous' Speed", 
                          ["Can we find speed at one exact moment?", 
                           "A single point makes our formula fail.", 
                           "We must shrink the interval to approach zero."])
        
        # --- Animation Content ---
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_tip": False})
        path = axes.plot(lambda x: 0.25 * x**2, color=WHITE)
        point_p = Dot(axes.c2p(2, 1), color=BLUE)
        
        # Use asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=WHITE)

        self.place_in_area(axes, 'B3', 'D5', scale_factor=0.6)
        self.place_at_grid(path, 'C4', scale_factor=0.6)
        self.place_at_grid(point_p, 'C3', scale_factor=0.6)
        self.place_at_grid(particle, 'C3', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(particle.animate.move_to(axes.c2p(3, 2.25)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Show the division by zero conceptually
        formula = MathTex(r"\frac{\Delta y}{\Delta x} = \frac{0}{0} = ?", color=RED)
        self.place_at_grid(formula, 'E5', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Visualize shrinking interval
        point_q = Dot(axes.c2p(3, 2.25), color=YELLOW)
        self.place_at_grid(point_q, 'D4', scale_factor=0.6)
        self.play(FadeIn(point_q))
        
        # Show path of shrinking
        self.play(point_q.animate.move_to(axes.c2p(2.2, 1.21)), run_time=2)
        self.wait(2)
