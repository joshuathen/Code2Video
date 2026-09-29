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
        self.setup_layout("The Geometric Bridge: Polar Coordinates", 
                          ["Square the integral to create a 2D plane.", 
                           "Switch to polar coordinates: r, dr, and dθ.", 
                           "The factor 'r' appears as the key."])
        
        # === Animation for Lecture Line 1 ===
        # Show 2D coordinate plane with axes.
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"color": "#808080"})
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.5)
        self.play(Create(axes))
        self.lecture[0].set_color("#808080")

        # === Animation for Lecture Line 2 ===
        # Draw a circle of radius r and add the bridge icon.
        circle = Circle(radius=1.5, color="#FFA500")
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color="#FFA500")
        label_r = MathTex("r", color="#FFA500")
        
        # Position circle and bridge
        self.place_in_area(circle, 'B3', 'E5', scale_factor=0.5)
        self.place_in_area(bridge_icon, 'C3', 'D4', scale_factor=0.3)
        label_r.next_to(circle.point_at_angle(PI/4), UR, buff=0.1)
        
        self.play(Create(circle), Write(label_r), FadeIn(bridge_icon))
        self.lecture[1].set_color("#FFA500")

        # === Animation for Lecture Line 3 ===
        # Show the transformation: dx dy = r dr dtheta.
        transform_text = MathTex(r"dx \\, dy = r \\, dr \\, d\\theta", color="#00FFFF")
        self.place_at_grid(transform_text, 'F5', scale_factor=0.9)
        
        self.play(Write(transform_text))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
