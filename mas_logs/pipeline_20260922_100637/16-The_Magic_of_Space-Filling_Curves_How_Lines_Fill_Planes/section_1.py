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
        lecture_lines = [
            "Can a one-dimensional line fill a square?",
            "Imagine a snake getting thinner and longer.",
            "It eventually blankets the entire surface area."
        ]
        self.setup_layout("The Paradox: Can a Line Fill a Square?", lecture_lines)
        
        # Load Asset
        snake_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snake.svg")
        
        # Setup visual elements
        square = Square(side_length=3, color=WHITE)
        self.place_in_area(square, 'B2', 'E5', scale_factor=0.6)
        
        self.place_in_area(snake_icon, 'B2', 'E5', scale_factor=0.4)
        
        line = Line(start=square.get_left(), end=square.get_right(), color=RED)
        line.move_to(square.get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(square), FadeIn(snake_icon), Create(line))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Create wavy path
        wavy_path = VGroup()
        for i in range(10):
            # Constraint: use place_in_area as per Issue 22
            segment = Line(start=square.get_left() + UP * (1.5 - i * 0.33), 
                           end=square.get_right() + UP * (1.5 - (i+1) * 0.33), 
                           color=GREEN)
            wavy_path.add(segment)
        
        self.place_in_area(wavy_path, 'B2', 'E5', scale_factor=0.75)
        
        self.play(Transform(line, wavy_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        # Grid visual as requested in Issue 23
        grid_visual = VGroup()
        for i in range(6):
            grid_visual.add(Line(self.grid[f'B{i+1}'], self.grid[f'F{i+1}'], color=BLUE))
        self.place_in_area(grid_visual, 'B2', 'F6', scale_factor=0.9)
        
        self.play(line.animate.set_color(BLUE), FadeIn(grid_visual), run_time=2)
        self.wait(2)
