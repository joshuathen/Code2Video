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
        self.setup_layout("Gradient Descent: The Path to Perfection", 
                          ["Loss functions form a complex, hilly landscape.", 
                           "Gradients act as a compass pointing downhill.", 
                           "Descending the slope lowers the error step-by-step."])
        
        # Define landscape
        axes = ThreeDAxes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[0, 2, 1], 
                          x_length=4, y_length=4, z_length=2)
        landscape = axes.plot_surface(
            lambda x, y: 0.5 * (np.sin(2 * x) + np.cos(2 * y)) + 1,
            u_range=[-2, 2], v_range=[-2, 2],
            resolution=(30, 30),
        )
        landscape.set_color(WHITE)
        landscape.set_opacity(0.5)
        
        # Use provided asset for the marble
        marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(landscape, "B2", "F5", scale_factor=0.6)
        self.play(Create(landscape))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(marble, "C3", scale_factor=0.5)
        marble.set_color("#FF0000")
        self.play(FadeIn(marble))
        self.lecture[1].set_color("#00FFFF") # Compass Blue

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        # Animate marble to a valley
        valley_pos = self.grid["E5"]
        self.play(marble.animate.move_to(valley_pos), run_time=2)
        marble.set_color("#00FF00")
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
