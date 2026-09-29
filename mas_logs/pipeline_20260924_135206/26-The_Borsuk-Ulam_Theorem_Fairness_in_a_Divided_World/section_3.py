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
        lecture_lines = ["Topology helps us solve discrete problems.", "Necklaces represent linear, ordered resource sets.", "Cuts divide these sets into fair shares."]
        self.setup_layout("Transition: From Geometry to Combinatorics", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset for necklace icon
        necklace_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")
        bead_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg")
        
        # Grid representation (as per feedback)
        sphere_grid = VGroup(*[Square(side_length=0.5, color="#BDC3C7") for _ in range(16)]).arrange_in_grid(4, 4, buff=0.1)
        self.place_in_area(sphere_grid, 'A4', 'C6', scale_factor=0.6)
        
        self.play(Create(sphere_grid))
        self.play(self.lecture[0].animate.set_color("#BDC3C7"))

        # === Animation for Lecture Line 2 ===
        # Use necklace asset as requested
        necklace_set = VGroup(necklace_asset)
        self.place_in_area(necklace_set, 'E1', 'E6', scale_factor=0.8)
        self.play(FadeIn(necklace_set))
        self.play(self.lecture[1].animate.set_color("#BDC3C7"))

        # === Animation for Lecture Line 3 ===
        # Cut line
        cut_line = Line(start=UP*0.5, end=DOWN*0.5, color="#FF0000")
        self.place_at_grid(cut_line, 'D3', scale_factor=0.9)
        
        # Bead indicator
        bead_indicator = bead_asset.copy().scale(0.5).set_color("#FFD700")
        self.place_at_grid(bead_indicator, 'D4', scale_factor=0.5)

        self.play(Create(cut_line), FadeIn(bead_indicator))
        self.play(self.lecture[2].animate.set_color("#E67E22"))
        
        self.wait(2)
