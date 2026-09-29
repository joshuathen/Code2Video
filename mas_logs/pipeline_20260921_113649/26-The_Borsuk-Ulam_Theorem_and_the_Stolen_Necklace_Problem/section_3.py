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
        self.setup_layout("The Stolen Necklace Problem", [
            "Thieves must divide a beaded necklace.",
            "Each needs half of every bead type.",
            "This is the discrete necklace problem."
        ])
        
        # Assets
        beads_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beads.svg")
        necklace_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")

        # === Animation for Lecture Line 1 ===
        # Using bead asset
        self.place_at_grid(beads_asset, 'B3', scale_factor=0.6)
        self.play(Create(beads_asset))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Create Thief A and B areas
        thief_a = Rectangle(width=2, height=1, color=GREEN).set_opacity(0.2)
        thief_b = Rectangle(width=2, height=1, color=ORANGE).set_opacity(0.2)
        thief_boxes = VGroup(thief_a, thief_b).arrange(RIGHT, buff=1.0)
        
        divider_lines = VGroup(Line(UP*0.5, DOWN*0.5, color=WHITE), Line(UP*0.5, DOWN*0.5, color=WHITE))
        
        # Fixes per VideoCritic
        self.place_in_area(necklace_asset, 'B1', 'B6', scale_factor=0.6)
        self.place_at_grid(thief_boxes, 'D2', scale_factor=0.7)
        self.place_at_grid(divider_lines, 'C1', scale_factor=0.8)
        
        self.play(FadeIn(thief_boxes), Create(divider_lines))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(ReplacementTransform(beads_asset, necklace_asset))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
