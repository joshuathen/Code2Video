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
        self.setup_layout("The Impossible Puzzle", [
            "Two blocks collide on a frictionless surface.",
            "A small mass hits a massive block.",
            "The big block then hits a wall.",
            "How many collisions happen in total?",
            "The count surprisingly reveals Pi digits."
        ])
        
        # Using SVG Assets
        puzzle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE)
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=GRAY)
        
        self.place_at_grid(puzzle, 'B2', scale_factor=0.5)
        
        physics = Text("Physics", color="#FF5733")
        self.place_at_grid(physics, 'B5', scale_factor=0.5)
        
        geometry = Text("Geometry", color="#33FF57")
        self.place_at_grid(geometry, 'E3', scale_factor=0.5)
        
        conclusion = Text("Conclusion", color="#3357FF")
        self.place_at_grid(conclusion, 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(puzzle))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(physics))
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(geometry))
        self.lecture[2].set_color("#33FF57")

        # === Animation for Lecture Line 4 ===
        # Animate puzzle vibrating near the wall
        self.add(wall)
        self.place_at_grid(wall, 'B3', scale_factor=0.5)
        self.play(puzzle.animate.shift(0.1 * UP), run_time=0.3)
        self.play(puzzle.animate.shift(0.1 * DOWN), run_time=0.3)
        self.lecture[3].set_color(WHITE)

        # === Animation for Lecture Line 5 ===
        self.play(Flash(conclusion, color="#3357FF", line_length=0.2, num_lines=12))
        self.lecture[4].set_color("#3357FF")
        self.wait(2)
