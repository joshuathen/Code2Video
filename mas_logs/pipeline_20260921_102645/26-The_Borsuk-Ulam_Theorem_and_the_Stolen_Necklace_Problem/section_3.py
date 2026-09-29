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
        lecture_lines = [
            "Thieves divide a necklace with different beads.",
            "They want an equal share of every bead type.",
            "We can cut the necklace at most k times."
        ]
        self.setup_layout("The Stolen Necklace Problem", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display necklace and beads using assets
        necklace = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")
        bead = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg")
        
        beads = VGroup(*[bead.copy() for _ in range(10)])
        beads.arrange(RIGHT, buff=0.1)
        self.place_in_area(beads, 'C4', 'C6', scale_factor=0.8)
        self.play(FadeIn(beads))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show two thieves dividing segments
        alice = Text("Alice", color=BLUE, font_size=20)
        bob = Text("Bob", color=GOLD, font_size=20)
        
        # Addressing issue 28/29: Reposition
        self.place_at_grid(alice, 'B3', scale_factor=0.7)
        self.place_at_grid(bob, 'E3', scale_factor=0.7)
        
        # Highlight segments (using bead asset)
        split_beads = VGroup(beads[:5].copy(), beads[5:].copy())
        self.play(
            FadeOut(beads),
            FadeIn(alice),
            FadeIn(bob),
            split_beads[0].animate.move_to(self.grid['B4']),
            split_beads[1].animate.move_to(self.grid['E4']),
            run_time=2
        )
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        # Animate cutting the necklace
        cut_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")
        cut = Line(UP*0.5, DOWN*0.5, color=RED)
        cut.move_to(self.grid['C3'])
        
        self.play(Create(cut), FadeIn(cut_icon.scale(0.5).move_to(self.grid['C3'])))
        self.lecture[2].set_color("#FF5733")
        self.wait(1)
