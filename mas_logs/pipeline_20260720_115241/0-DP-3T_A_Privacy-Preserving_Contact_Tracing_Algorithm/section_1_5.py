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

class Section15Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Cryptography keeps data secret and tamper-proof.",
            "Symmetric encryption uses shared secret keys.",
            "Hashing creates unique fingerprints for data.",
            "These tools protect information in DP-3T.",
            "No complex math needed for understanding."
        ]
        self.setup_layout("Prerequisite: Basic Cryptography Concepts", lecture_lines)

        # Colors for lecture lines and corresponding animations
        colors = [
            "#FFD700",  # Gold for Lecture Line 1
            "#FF6347",  # Tomato for Lecture Line 2
            "#87CEEB",  # SkyBlue for Lecture Line 3
            "#32CD32",  # LimeGreen for Lecture Line 4
            "#DA70D6"   # Orchid for Lecture Line 5
        ]

        # === Animation for Lecture Line 1: Cryptography keeps data secret and tamper-proof. ===
        # Representing "secret" with a locked box
        locked_box = Rectangle(height=1.5, width=1.5, color=colors[0])
        locked_box.set_fill(color=BLACK, opacity=1)
        lock_symbol = VGroup(
            Circle(radius=0.1, color=WHITE),
            Line(LEFT*0.1, RIGHT*0.1, color=WHITE),
            Line(UP*0.1, DOWN*0.1, color=WHITE)
        ).move_to(locked_box.get_center() + UP*0.2)
        locked_box_with_lock = VGroup(locked_box, lock_symbol)
        self.play(FadeIn(locked_box_with_lock, shift=DOWN))
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.wait(1)
        self.play(FadeOut(locked_box_with_lock, shift=UP))
        self.play(self.lecture[0].animate.set_color(WHITE)) # Reset color

        # === Animation for Lecture Line 2: Symmetric encryption uses shared secret keys. ===
        # Representing shared secret keys
        # FIX: Define Key class or use a suitable Manim object.
        # Assuming 'Key' is a custom Mobject, we'll define a simple one for demonstration.
        # In a real scenario, this would likely be imported or defined elsewhere.
        class Key(SVGMobject):
            def __init__(self, file_name, **kwargs): # Changed svg_file to file_name
                super().__init__(file_name=file_name, **kwargs) # Changed svg_file to file_name

        # If an SVG file is not available, a simple shape can be used as a placeholder:
        # class Key(Polygon):
        #     def __init__(self, **kwargs):
        #         points = [
        #             UP * 0.5,
        #             LEFT * 0.2 + DOWN * 0.5,
        #             RIGHT * 0.2 + DOWN * 0.5,
        #             UP * 0.5,
        #             UP * 0.5 + RIGHT * 0.1,
        #             UP * 0.2 + RIGHT * 0.1,
        #             UP * 0.2 + LEFT * 0.1,
        #             UP * 0.5 + LEFT * 0.1,
        #         ]
        #         super().__init__(*points, **kwargs)
        
        # FIX: Pass the file_name argument to the Key constructor.
        key1 = Key(file_name="/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg", color=colors[1])
        key2 = Key(file_name="/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg", color=colors[1])
        key1.scale(0.8)
        key2.scale(0.8)
        
        self.play(FadeIn(key1.move_to(self.grid["B3"]), shift=DOWN))
        self.play(FadeIn(key2.move_to(self.grid["C3"]), shift=DOWN))
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.wait(1)
        self.play(FadeOut(key1, key2, shift=UP))
        self.play(self.lecture[1].animate.set_color(WHITE)) # Reset color

        # === Animation for Lecture Line 3: Hashing creates unique fingerprints for data. ===
        # Representing data and its hash fingerprint
        data_block = Square(side_length=1.2, color=colors[2]).move_to(self.grid["D2"])
        data_block.set_fill(color=BLACK, opacity=1)
        fingerprint_icon = VGroup(
            Line(UP*0.5, DOWN*0.5, color=WHITE),
            Line(LEFT*0.5, RIGHT*0.5, color=WHITE)
        ).scale(0.8).next_to(data_block, RIGHT, buff=0.5)
        
        self.play(FadeIn(data_block, shift=DOWN))
        self.play(FadeIn(fingerprint_icon, shift=DOWN))
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.wait(1)
        self.play(FadeOut(data_block, fingerprint_icon, shift=UP))
        self.play(self.lecture[2].animate.set_color(WHITE)) # Reset color

        # === Animation for Lecture Line 4: These tools protect information in DP-3T. ===
        # Representing DP-3T with a shield
        shield = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shield.svg", color=colors[3]) # Placeholder for actual asset
        # For demonstration without actual asset, use a simple shape
        shield_placeholder = Polygon(
            *[
                UP*1, 
                LEFT*0.5 + DOWN*0.5, 
                RIGHT*0.5 + DOWN*0.5
            ], 
            color=colors[3], 
            stroke_width=3
        )
        shield_placeholder.scale(1.2)
        self.play(FadeIn(shield_placeholder.move_to(self.grid["E4"]), shift=DOWN))
        self.play(self.lecture[3].animate.set_color(colors[3]))
        self.wait(1)
        self.play(FadeOut(shield_placeholder, shift=UP))
        self.play(self.lecture[3].animate.set_color(WHITE)) # Reset color

        # === Animation for Lecture Line 5: No complex math needed for understanding. ===
        # Representing simplicity with a checkmark
        checkmark = VGroup(
            Line(LEFT*0.5, ORIGIN, stroke_width=3),
            Line(ORIGIN, DOWN*0.5, stroke_width=3)
        ).scale(1.5).rotate(PI/4).set_color(colors[4])
        self.play(FadeIn(checkmark.move_to(self.grid["F5"]), shift=DOWN))
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.wait(1)
        self.play(FadeOut(checkmark, shift=UP))
        self.play(self.lecture[4].animate.set_color(WHITE)) # Reset color

        self.wait(2)

# Note: For actual asset usage, replace "/scratch/pawsey1357/jthen/Code2Video/assets/icon/shield.svg" with the correct path.
# The Key class is a placeholder; in a real scenario, you might define a Key class
# or use a pre-made SVGMobject if available.