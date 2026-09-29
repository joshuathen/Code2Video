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
            "The warden marks a secret square on the board.",
            "Coins cover the board randomly, heads or tails.",
            "One prisoner flips a coin to signal the square."
        ]
        self.setup_layout("Introduction: The Prisoner's Dilemma", lecture_lines)
        
        # Assets
        warden_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/warden.svg")
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        prisoner_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prisoner.svg")
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Create a text label "The Warden's Mark" in #FFFFFF at center using warden.svg.
        warden_text = Text("The Warden's Mark", font_size=36, color=WHITE)
        self.place_at_grid(warden_text, 'C1', scale_factor=0.8)
        self.place_at_grid(warden_icon, 'C3', scale_factor=0.5)
        self.play(Write(warden_text), FadeIn(warden_icon))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Fade in a 8x8 grid populated with coins.
        self.place_in_area(grid_icon, 'A4', 'F6', scale_factor=0.8)
        self.play(FadeIn(grid_icon))
        self.lecture[1].set_color("#4DA6FF")
        
        # === Animation for Lecture Line 3 ===
        # Highlight one cell with a distinct color #FF0000 representing the prisoner.
        self.place_at_grid(prisoner_icon, 'B4', scale_factor=0.6)
        prisoner_icon.set_color("#FF0000")
        self.play(FadeIn(prisoner_icon))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
