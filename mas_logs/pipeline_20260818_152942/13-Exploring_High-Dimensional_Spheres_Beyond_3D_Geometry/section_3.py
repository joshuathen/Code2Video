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
        self.setup_layout("The 'Curse' and 'Blessing' of Dimensionality", [
            "N-balls and hypercubes exhibit different behaviors.",
            "In high dimensions, spheres lose space in corners.",
            "The center becomes a void for high-dimensional objects."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show data sparsity: A cube and a sphere
        cube = Cube(side_length=1.5, fill_opacity=0.3, color="#FFA500")
        sphere = Sphere(radius=0.7, fill_opacity=0.6, color="#FFA500")
        group1 = VGroup(cube, sphere)
        # Using place_in_area as per feedback #28/37
        self.place_in_area(group1, 'B4', 'C6', scale_factor=1.0)
        self.play(Create(cube), GrowFromCenter(sphere))
        self.lecture[0].set_color("#FFA500")

        # === Animation for Lecture Line 2 ===
        # Highlight distance concentration: Heatmap/dots in corners
        dots = VGroup(*[Dot(point=self.grid['C4'] + np.array([x, y, 0]) * 0.3, color="#FF4500") 
                        for x in [-1, 1] for y in [-1, 1]])
        self.play(FadeIn(dots))
        self.lecture[1].set_color("#FF4500")

        # === Animation for Lecture Line 3 ===
        # Balance scale asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg
        scale = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg").set_color("#7FFF00")
        # Position using place_at_grid as per feedback #27/37
        self.place_at_grid(scale, 'E2', scale_factor=0.7)
        self.play(FadeIn(scale))
        self.lecture[2].set_color("#7FFF00")
        self.wait(2)
