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
        self.setup_layout("Compressing Dimensions (m < n)", [
            "Projection collapses high to low dimensions.",
            "Imagine casting a 3D shadow onto 2D.",
            "Information is lost during this compression."
        ])
        
        # Visual elements
        cube = Cube(side_length=1.5, fill_opacity=0.3, stroke_width=2)
        # Using SVGMobject as requested for the plane
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/screen.svg", fill_opacity=0.5, stroke_width=2, color=BLUE)
        
        # === Animation for Lecture Line 1 ===
        # Projection collapses high to low dimensions.
        self.lecture[0].set_color("#FFFFFF")
        # Placing cube in a balanced position
        self.place_in_area(cube, 'A3', 'C4', scale_factor=0.6)
        self.play(Create(cube))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine casting a 3D shadow onto 2D.
        self.lecture[1].set_color("#FF8080")
        # Placing plane in a balanced position
        self.place_in_area(plane, 'D3', 'F4', scale_factor=0.5)
        shadow = Rectangle(width=1.0, height=1.0, fill_opacity=0.5, color=RED)
        shadow.move_to(plane.get_center())
        
        self.play(Create(plane))
        self.play(TransformFromCopy(cube, shadow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Information is lost during this compression.
        self.lecture[2].set_color("#80FF80")
        loss_text = Text("Loss of depth", font_size=20, color=YELLOW)
        # Positioning at D4 as requested
        self.place_at_grid(loss_text, 'D4', scale_factor=0.9)
        
        self.play(FadeIn(loss_text))
        self.play(Indicate(shadow))
        self.wait(2)
