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
        # Assets
        warrior_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/warrior.svg"
        potion_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/potion.svg"
        
        # Setup
        title = "Abstract Example 2: The Character Stat Vector"
        lines = [
            "Character stats in games behave like vector spaces.",
            "A Berserker Potion scales strength and speed attributes.",
            "Combining character classes represents simple vector addition."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a text vector [Strength: 10, Speed: 5] #FFFFFF labeled 'Warrior' [Asset: warrior.svg]
        self.lecture[0].set_color(WHITE)
        
        warrior_icon = SVGMobject(warrior_path)
        self.place_at_grid(warrior_icon, "B3", scale_factor=0.6)
        
        warrior_label = Text("Warrior", color=WHITE, font_size=24)
        self.place_at_grid(warrior_label, "A3")
        
        warrior_stats = Text("[Strength: 10, Speed: 5]", color=WHITE, font_size=20)
        self.place_at_grid(warrior_stats, "C3")
        
        self.play(
            FadeIn(warrior_icon),
            Write(warrior_label),
            Write(warrior_stats)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # The vector's values multiply by 2 using [Asset: potion.svg] 
        # to become [Strength: 20, Speed: 10] with label 'Berserker' #FF0000.
        self.lecture[1].set_color(RED)
        
        potion_icon = SVGMobject(potion_path)
        self.place_at_grid(potion_icon, "C5", scale_factor=0.5)
        
        scalar_text = Text("x 2.0", color=RED, font_size=28)
        self.place_at_grid(scalar_text, "B5")
        
        berserker_label = Text("Berserker", color=RED, font_size=24)
        self.place_at_grid(berserker_label, "A3")
        
        berserker_stats = Text("[Strength: 20, Speed: 10]", color=RED, font_size=20)
        self.place_at_grid(berserker_stats, "C3")
        
        self.play(FadeIn(potion_icon), Write(scalar_text))
        self.play(
            ReplacementTransform(warrior_label, berserker_label),
            ReplacementTransform(warrior_stats, berserker_stats),
            warrior_icon.animate.set_color(RED),
            potion_icon.animate.scale(1.2).fade(1)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Warrior vector + Mage vector [Strength: 2, Speed: 8] #00FFFF = Paladin vector [Strength: 12, Speed: 13] #FFFF00.
        self.lecture[2].set_color(YELLOW)
        
        # Clear previous for new layout
        self.play(
            FadeOut(potion_icon),
            FadeOut(scalar_text),
            FadeOut(berserker_label),
            FadeOut(berserker_stats),
            FadeOut(warrior_icon)
        )
        
        # Layout for Addition
        # Warrior (Col 2)
        w_label = Text("Warrior", color=WHITE, font_size=18)
        self.place_at_grid(w_label, "A1")
        w_vec = Text("[10, 5]", color=WHITE, font_size=20)
        self.place_at_grid(w_vec, "B1")
        
        plus = Text("+", font_size=30)
        self.place_at_grid(plus, "B2")
        
        # Mage (Col 3)
        m_label = Text("Mage", color=TEAL, font_size=18)
        self.place_at_grid(m_label, "A3")
        m_vec = Text("[2, 8]", color=TEAL, font_size=20)
        self.place_at_grid(m_vec, "B3")
        
        equals = Text("=", font_size=30)
        self.place_at_grid(equals, "B4")
        
        # Paladin (Col 5)
        p_label = Text("Paladin", color=YELLOW, font_size=18)
        self.place_at_grid(p_label, "A5")
        p_vec = Text("[12, 13]", color=YELLOW, font_size=20)
        self.place_at_grid(p_vec, "B5")
        
        self.play(
            Write(w_label), Write(w_vec),
            Write(plus),
            Write(m_label), Write(m_vec)
        )
        self.play(
            Write(equals),
            Write(p_label), Write(p_vec)
        )
        self.wait(3)
