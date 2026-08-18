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
        self.setup_layout("Intuition: From 2D to 3D", ["We start with a 2D circle.", "Transition to a 3D sphere.", "Slicing reveals inner geometry."])
        
        # Using SVG asset for 2D circle and 3D sphere
        shape_2d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FF5733", fill_opacity=0.5)
        self.place_at_grid(shape_2d, 'C2', scale_factor=0.8)
        
        shape_3d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#33FF57", fill_opacity=0.5)
        # Fix 1 & 2: Adjust position of 3D object to avoid overlap
        self.place_at_grid(shape_3d, 'C5', scale_factor=0.8)
        
        # Labels
        label_2d = Text("2D Circle", font_size=20)
        # Fix 3: Adjust position of labels
        self.place_at_grid(label_2d, 'B2', scale_factor=0.9)
        
        label_3d = Text("3D Sphere", font_size=20)
        self.place_at_grid(label_3d, 'B5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(shape_2d), FadeIn(label_2d))
        self.play(self.lecture[0].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(shape_3d), FadeIn(label_3d))
        self.play(self.lecture[1].animate.set_color("#33FF57"))

        # === Animation for Lecture Line 3 ===
        slice_plane = Rectangle(width=1.5, height=1.5, color="#FFFFFF", fill_opacity=0.5)
        # Fix 4: Place slice plane at E5
        self.place_at_grid(slice_plane, 'E5', scale_factor=0.7)
        self.play(FadeIn(slice_plane))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(1)
