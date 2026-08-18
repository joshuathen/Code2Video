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

class Section1Scene(MovingCameraScene, TeachingScene):
    def construct(self):
        lecture_lines = [
            "Digital currency faces a double spending problem.",
            "A shared ledger records all transactions securely.",
            "Transactions are organized into chronological blocks."
        ]
        self.setup_layout("Introduction: The Digital Ledger", lecture_lines)
        
        # Visual assets
        ledger_text = Text("Digital Ledger", font_size=36, color=WHITE)
        # Using SVG asset
        chain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chain.svg", color=BLUE)
        chain_group = VGroup(*[chain_icon.copy() for _ in range(3)]).arrange(RIGHT, buff=0.2)
        
        # === Animation for Lecture Line 1 ===
        # Fade in the text 'Digital Ledger'.
        self.place_at_grid(ledger_text, 'A3', scale_factor=0.8)
        self.play(FadeIn(ledger_text))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/chain.svg] graphic connecting blocks.
        self.place_in_area(chain_group, 'B3', 'E5', scale_factor=1.0)
        self.play(FadeIn(chain_group))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight chain nodes with #FF5733.
        # Zoom out to show the full structure [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/chain.svg].
        self.play(
            *[obj.animate.set_color("#FF5733") for obj in chain_group],
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(self.camera.frame.animate.scale(1.2))
        self.wait(2)
