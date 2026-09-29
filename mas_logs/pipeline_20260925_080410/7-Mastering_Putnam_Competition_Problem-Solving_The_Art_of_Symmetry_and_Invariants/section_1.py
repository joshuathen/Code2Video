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
            "Putnam problems demand creative, first-principle thinking.",
            "Invariants remain unchanged despite system evolution.",
            "Consider a chameleon moving on a checkerboard.",
            "Parity shows the square's color invariant.",
            "This core concept drives Putnam problem-solving."
        ]
        self.setup_layout("Introduction: The Putnam Mindset", lecture_lines)
        
        # Load Assets
        bulb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg")
        maze = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg")
        chameleon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(FadeIn(self.place_at_grid(bulb, 'A6', scale_factor=0.6)))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), self.lecture[1].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(FadeIn(self.place_in_area(maze, 'D4', 'F6', scale_factor=0.4)))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]), self.lecture[3].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.play(FadeIn(self.place_at_grid(chameleon, 'C3', scale_factor=0.5)))
        self.play(Indicate(self.lecture[4]))
