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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Flatland Foundation", [
            "Flatland beings perceive 3D objects as shifting 2D slices.",
            "Imagine a sphere passing through a 2D plane.",
            "It starts as a point, expands, then shrinks."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in Point using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg], color #FFFFFF, label "Point"
        point = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FFFFFF")
        label_point = Text("Point", font_size=20, color="#FFFFFF")
        self.place_at_grid(point, 'B3', scale_factor=0.7)
        self.place_at_grid(label_point, 'B4', scale_factor=0.7)
        self.play(FadeIn(point), Write(label_point))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Morph Point into Line, color #00FF00, label "Line segment"
        # Rotate Line segment in 2D space, color #00FFFF
        line = Line(start=np.array([0, 0, 0]), end=np.array([1, 0, 0]), color="#00FF00")
        label_line = Text("Line segment", font_size=20, color="#00FF00")
        self.place_at_grid(line, 'C3', scale_factor=0.8)
        self.place_at_grid(label_line, 'C4', scale_factor=0.8)
        self.play(ReplacementTransform(point, line), ReplacementTransform(label_point, label_line))
        self.play(Rotate(line, angle=PI/4), run_time=1)
        line.set_color("#00FFFF")
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Show 2D grid appearance, color #FF00FF, label "Flatland Plane"
        # Highlight Flatland within the 2D grid using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg], color #FFFF00
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], color="#FF00FF")
        label_grid = Text("Flatland Plane", font_size=20, color="#FF00FF")
        self.place_in_area(grid, 'D2', 'F5', scale_factor=0.4)
        self.place_at_grid(label_grid, 'D2', scale_factor=0.6)
        self.play(Create(grid), Write(label_grid))
        
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FFFF00")
        self.place_at_grid(sphere_asset, 'E3', scale_factor=0.5)
        self.play(Create(sphere_asset))
        
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)
