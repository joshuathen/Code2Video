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
        self.setup_layout("The Stolen Necklace Problem", [
            "Two thieves share loot.", 
            "Cut necklaces perfectly now.", 
            "Fair division uses cuts.", 
            "Borsuk-Ulam guarantees equitable splits.", 
            "Optimal cuts divide necklaces."
        ])
        
        # Load assets
        beads_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beads.svg")
        necklace_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")
        
        # Setup visual elements
        necklace_full = beads_icon.copy()
        
        # Apply positioning from critic/issue feedback
        self.place_in_area(necklace_full, 'A2', 'A5', scale_factor=0.9)
        
        thief_labels = VGroup(
            Text("Thief 1", color=BLUE, font_size=24), 
            Text("Thief 2", color=RED, font_size=24)
        ).arrange(RIGHT, buff=1.0)
        
        self.place_in_area(thief_labels, 'E2', 'F5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(necklace_full), Write(thief_labels))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        cuts = VGroup(*[DashedLine(UP*0.5, DOWN*0.5, color="#FF4500") for _ in range(2)])
        cuts.arrange(RIGHT, buff=0.5).move_to(necklace_full.get_center())
        self.play(Create(cuts))
        self.lecture[1].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(necklace_full.animate.shift(DOWN*0.5))
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#32CD32")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        checkmark = Tex(r"$\checkmark$", color="#32CD32", font_size=72)
        self.place_at_grid(checkmark, 'D4', scale_factor=0.6)
        
        # Display final outcome
        final_necklace = necklace_icon.copy()
        self.place_in_area(final_necklace, 'C3', 'D4', scale_factor=0.5)
        
        self.play(Write(checkmark), FadeIn(final_necklace))
        self.lecture[4].set_color("#32CD32")
        self.wait(2)
