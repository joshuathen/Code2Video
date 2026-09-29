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
        lecture_lines = ["How do we unfold a 4D tesseract?", "We can flatten it into 3D.", "This creates the famous Dalí Cross."]
        self.setup_layout("Solving the Shift: The Unfolding Puzzle", lecture_lines)
        
        # Assets
        tesseract_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tesseract.svg")
        cube_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        # Elements
        cubes = VGroup(*[cube_svg.copy() for _ in range(8)])
        self.place_in_area(cubes, 'C3', 'E5', scale_factor=0.7)
        
        tesseract_animation = tesseract_svg
        self.place_in_area(tesseract_animation, 'C3', 'F5', scale_factor=0.65)
        
        dalí_cross_label = Text("Dalí Cross", font_size=24, color=YELLOW)
        self.place_at_grid(dalí_cross_label, 'D4', scale_factor=0.9)
        
        # Animation
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(tesseract_animation))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(
            FadeOut(tesseract_animation),
            FadeIn(cubes),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(
            FadeIn(dalí_cross_label),
            run_time=2
        )
        self.wait(2)
