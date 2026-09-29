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
        self.setup_layout("The Landscape of Loss", [
            "Visualize error as a 3D landscape.",
            "Weights are coordinates on this map.",
            "Bottom of the valley is perfect accuracy."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualize error as a 3D landscape.
        self.lecture[0].set_color("#00FFFF")
        
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[0, 2])
        bowl = Surface(
            lambda u, v: np.array([u, v, u**2 + v**2]),
            u_range=[-1.5, 1.5], v_range=[-1.5, 1.5],
            resolution=(20, 20)
        ).set_fill(color="#00FFFF", opacity=0.5).set_stroke(color="#FFFFFF", width=0.5)
        
        mountain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        
        axes_bowl_group = VGroup(axes, bowl, mountain_icon)
        
        # Using place_in_area for better positioning (Issue 24)
        self.place_in_area(axes_bowl_group, 'C4', 'E6', scale_factor=0.5)
        self.play(Create(axes), Create(bowl), FadeIn(mountain_icon))

        # === Animation for Lecture Line 2 ===
        # Weights are coordinates on this map.
        self.lecture[1].set_color("#FFFF00")
        
        point = Dot(color="#FFFF00")
        point_label = Text('Current Weights', font_size=16, color="#FFFF00")
        
        # Using place_at_grid for point and label (Issue 25, 26)
        self.place_at_grid(point, 'D5', scale_factor=0.6)
        self.place_at_grid(point_label, 'D4', scale_factor=0.4)
        
        valley_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        self.place_at_grid(valley_icon, 'E5', scale_factor=0.3)
        
        self.play(FadeIn(point), FadeIn(point_label), FadeIn(valley_icon))

        # === Animation for Lecture Line 3 ===
        # Bottom of the valley is perfect accuracy.
        self.lecture[2].set_color("#00FF00")
        self.play(point.animate.move_to(axes.c2p(0, 0, 0)), run_time=2)
        self.wait(1)
