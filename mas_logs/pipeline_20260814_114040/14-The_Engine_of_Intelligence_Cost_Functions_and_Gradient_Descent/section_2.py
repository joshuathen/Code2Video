from manim import *
import numpy as np

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
        self.setup_layout("Visualizing the Terrain: The Loss Landscape", 
                          ["Visualize cost as a 3D mountain range.", 
                           "Each point represents a network's performance.", 
                           "We seek the lowest valley for accuracy."])
        
        # === Animation for Lecture Line 1 ===
        # Create a simple 3D landscape using a Surface
        axes = ThreeDAxes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[0, 2, 1], 
                          x_length=3, y_length=3, z_length=1.5)
        
        def loss_function(u, v):
            return np.array([u, v, 0.5 * (u**2 + v**2)])
        
        surface = Surface(loss_function, u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], 
                          resolution=(20, 20), color=BLUE_D, fill_opacity=0.6)
        
        landscape = VGroup(axes, surface)
        # Apply fix: use C3-F6 and scale factor 0.45
        self.place_in_area(landscape, "C3", "F6", scale_factor=0.45)
        self.play(Create(landscape), self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # Show the current performance as a point
        point = Dot(color=YELLOW).scale(1.5)
        # Place point relative to axes
        point.move_to(axes.c2p(1.2, 1.2, 0.5 * (1.2**2 + 1.2**2)))
        
        self.add(point)
        self.play(FadeIn(point), self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        # Highlight the lowest valley (origin 0,0,0)
        target = Dot(axes.c2p(0, 0, 0), color=GREEN).scale(2)
        
        self.play(FadeIn(target), self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
