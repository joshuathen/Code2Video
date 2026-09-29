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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: The Final Synthesis", [
            "Simplify, Explore Invariants, Apply Extremes, Prove.",
            "Treat Putnam problems as puzzles, not tests.",
            "Integrate your logic engine for synthesis."
        ])
        
        # Load SVG
        puzzle_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg").set_color("#32CD32")
        
        # Pieces for the Jigsaw Puzzle
        piece1 = Text("Intuition", font_size=24, color="#9370DB")
        piece2 = Text("Logic", font_size=24, color="#9370DB")
        piece3 = Text("Proof", font_size=24, color="#9370DB")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(piece1, "C2", scale_factor=0.7)
        self.play(FadeIn(piece1))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        self.place_at_grid(piece2, "C5", scale_factor=0.7)
        self.place_at_grid(piece3, "E4", scale_factor=0.7)
        self.play(FadeIn(piece2), FadeIn(piece3))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        
        # Assemble logic engine puzzle
        puzzle_group = VGroup(puzzle_svg, piece1, piece2, piece3)
        self.place_in_area(puzzle_group, "D1", "F6", scale_factor=0.9)
        
        self.play(
            FadeIn(puzzle_svg),
            puzzle_group.animate.set_color(WHITE),
            run_time=2
        )
        self.wait(2)
