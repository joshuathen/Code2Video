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
        # Setup the scene layout
        title_str = "Visualizing the Magnitude of 2^256"
        lecture_lines = [
            "Imagine one grain of sand represents a single hash.",
            "All sand on Earth is just ten to the eighteenth.",
            "Our body holds even more atoms, ten to twenty-seventh.",
            "Yet, two to the two-fifty-six is ten to seventy-seven.",
            "That is nearly every atom in the observable universe."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Hexadecimal Colors as per L008
        GOLD_COLOR = "#FFD700"
        SAND_COLOR = "#EEDC82"
        EARTH_COLOR = "#4B9CD3"
        BODY_COLOR = "#FFB6C1"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line: "Imagine one grain of sand represents a single hash."
        # Storyboard: Display the number '2^256' in gold #FFD700.
        self.lecture[0].set_color(SAND_COLOR)
        
        # Displaying 2^256 as the grand total
        self.val_256 = MathTex(r"2^{256}", color=GOLD_COLOR)
        self.place_at_grid(self.val_256, "B4", scale_factor=1.2)
        
        # Representation of a single hash as a grain of sand
        sand_grain = Dot(color=SAND_COLOR, radius=0.08)
        sand_label = Text("1 Hash", font_size=20, color=SAND_COLOR)
        sand_group = VGroup(sand_grain, sand_label).arrange(DOWN, buff=0.2)
        self.place_at_grid(sand_group, "C4", scale_factor=0.8) # L002: Scale 0.8

        self.play(FadeIn(self.val_256), FadeIn(sand_group))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line: "All sand on Earth is just ten to the eighteenth."
        # Storyboard: Scale down to show '10^18' grains of sand [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg] on Earth.
        self.lecture[1].set_color(EARTH_COLOR)

        # Label and representation for sand on Earth
        # Issue 32: place_at_grid(self.earth_label, 'D4', scale_factor=0.7)
        self.earth_label = MathTex(r"10^{18} \text{ grains}", color=EARTH_COLOR)
        self.place_at_grid(self.earth_label, "D4", scale_factor=0.7)
        
        # Using the requested Asset (Issue 17)
        self.earth_sand_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg")
        self.earth_sand_asset.set_color(EARTH_COLOR)
        self.place_at_grid(self.earth_sand_asset, "D5", scale_factor=0.8)
        
        # Visual Earth indicator
        earth_sphere = Circle(radius=0.4, color=EARTH_COLOR, fill_opacity=0.3)
        self.place_at_grid(earth_sphere, "D5", scale_factor=1.0)
        
        self.play(FadeIn(self.earth_label), FadeIn(self.earth_sand_asset), Create(earth_sphere))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Line: "Our body holds even more atoms, ten to twenty-seventh."
        # Storyboard: Show '10^27' atoms in the human body.
        self.lecture[2].set_color(BODY_COLOR)

        # Atoms in human body
        # Issue 32: place_at_grid(self.body_label, 'E4', scale_factor=0.7)
        self.body_label = MathTex(r"10^{27} \text{ atoms}", color=BODY_COLOR)
        self.place_at_grid(self.body_label, "E4", scale_factor=0.7)
        
        # Simple person figure built from mobjects
        p_head = Circle(radius=0.15, color=BODY_COLOR)
        p_torso = Line(p_head.get_bottom(), p_head.get_bottom() + DOWN*0.5, color=BODY_COLOR)
        p_arms = Line(p_torso.get_center() + LEFT*0.25, p_torso.get_center() + RIGHT*0.25, color=BODY_COLOR)
        p_leg1 = Line(p_torso.get_bottom(), p_torso.get_bottom() + DOWN*0.4 + LEFT*0.15, color=BODY_COLOR)
        p_leg2 = Line(p_torso.get_bottom(), p_torso.get_bottom() + DOWN*0.4 + RIGHT*0.15, color=BODY_COLOR)
        person_vgroup = VGroup(p_head, p_torso, p_arms, p_leg1, p_leg2)
        self.place_at_grid(person_vgroup, "E5", scale_factor=0.7)
        
        self.play(FadeIn(self.body_label), FadeIn(person_vgroup))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Line: "Yet, two to the two-fifty-six is ten to seventy-seven."
        # Storyboard: Zoom out to the observable universe.
        self.lecture[3].set_color(GOLD_COLOR)

        # The core comparison
        # Issue 31: place_in_area(self.ten_77_label, 'B2', 'B6', scale_factor=1.0)
        self.ten_77_label = MathTex(r"2^{256} \approx 10^{77} \text{ hashes}", color=GOLD_COLOR)
        self.place_in_area(self.ten_77_label, "B2", "B6", scale_factor=1.0)
        
        # Simulating "zoom out" by removing local objects and emphasizing the large number
        self.play(
            FadeOut(sand_group),
            FadeOut(self.earth_label),
            FadeOut(self.earth_sand_asset),
            FadeOut(earth_sphere),
            FadeOut(self.body_label),
            FadeOut(person_vgroup),
            FadeOut(self.val_256),
            FadeIn(self.ten_77_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Line: "That is nearly every atom in the observable universe."
        # Storyboard: Label the universe with '10^80 atoms' in white #FFFFFF.
        self.lecture[4].set_color(WHITE_COLOR)
        
        # Universe scale label
        # Issue 33: place_in_area(self.universe_atoms, 'D2', 'F6', scale_factor=0.9)
        self.universe_atoms = MathTex(r"10^{80} \text{ atoms in universe}", color=WHITE_COLOR)
        self.place_in_area(self.universe_atoms, "D2", "F6", scale_factor=0.9)
        
        # Star field simulation (limited count to avoid performance issues per L027)
        import random
        random.seed(42) # Deterministic randomness for MAS
        stars = VGroup(*[
            Dot(radius=0.015, color=WHITE_COLOR).move_to([
                random.uniform(0.5, 5.5), 
                random.uniform(-2.8, 2.2), 
                0
            ]) for _ in range(80)
        ])
        
        self.play(
            FadeIn(stars),
            FadeIn(self.universe_atoms),
            Indicate(self.ten_77_label, color=GOLD_COLOR) # L004: Highlight the comparison
        )
        self.wait(1.5)
