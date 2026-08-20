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
        self.setup_layout("The Familiar World: Real Analysis", [
            "Standard Euclidean distance measures physical separation.",
            "Sequences shrink as they approach a limit.",
            "A rabbit jumps half distance to a carrot."
        ])
        
        # === Animation for Lecture Line 1 ===
        number_line = NumberLine(x_range=[-1, 3, 1], length=5, include_numbers=True)
        rabbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg")
        self.place_in_area(number_line, 'D3', 'F6', scale_factor=0.9)
        self.place_at_grid(rabbit, 'A3', scale_factor=0.5)
        self.play(Create(number_line), FadeIn(rabbit))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        highlight = SurroundingRectangle(number_line.numbers[1:3], color="#FFFF00", buff=0.1)
        self.play(Create(highlight))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        label = Text("Absolute Value", font_size=24, color="#00FFFF")
        # Use ImageMobject for raster files like .png, as SVGMobject is for SVG/Vector files
        carrot = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/carrot.png")
        self.place_at_grid(label, 'B4', scale_factor=0.7)
        self.place_at_grid(carrot, 'C4', scale_factor=0.5)
        self.play(FadeIn(label), FadeIn(carrot))
        self.lecture[2].set_color("#00FFFF")
        
        self.wait(2)
