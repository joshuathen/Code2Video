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

class Section6Scene(TeachingScene):
    def construct(self):
        # Fetch the title and lines from the storyboard
        title_text = "Conclusion: The Gold Standard of Security"
        lecture_lines = [
            "SHA-256 is the foundation of modern digital security.",
            "Security depends on the vastness of the math space.",
            "This mathematical fortress remains uncrackable by any means."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        bitcoin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bitcoin.svg"
        padlock_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/padlock.svg"

        # === Animation for Lecture Line 1 ===
        # White icons for Bitcoin [Asset: ...] and a padlock [Asset: ...] (#FFFFFF) appear.
        self.lecture[0].set_color(WHITE)
        
        # Pre-load SVGs
        bitcoin_svg = SVGMobject(bitcoin_path, color=WHITE)
        padlock_svg = SVGMobject(padlock_path, color=WHITE)
        
        bitcoin_icon = bitcoin_svg.copy().scale(0.8)
        padlock_icon = padlock_svg.copy().scale(0.8)
        
        self.place_at_grid(bitcoin_icon, "B2")
        self.place_at_grid(padlock_icon, "B5")
        
        self.play(FadeIn(bitcoin_icon), FadeIn(padlock_icon))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # The blue detective (#ADD8E6) locks a digital chest.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#ADD8E6")
        
        # Fade out line 1 main icons to focus on chest
        self.play(FadeOut(bitcoin_icon), FadeOut(padlock_icon))

        # Detective (blue figure)
        detective_head = Circle(radius=0.15, color="#ADD8E6", fill_opacity=1, stroke_width=0)
        detective_body = Triangle(color="#ADD8E6", fill_opacity=1).scale(0.35).next_to(detective_head, DOWN, buff=0)
        detective = VGroup(detective_head, detective_body)
        detective_label = Text("Detective", font_size=18, color="#ADD8E6")
        self.place_at_grid(detective, "D2")
        detective_label.next_to(detective, DOWN, buff=0.1)
        
        # Digital Chest (white)
        chest_box = Rectangle(width=0.9, height=0.6, color=WHITE, fill_opacity=0.3, stroke_width=2)
        chest_lid = Line(chest_box.get_left() + UP*0.1, chest_box.get_right() + UP*0.1, color=WHITE)
        chest = VGroup(chest_box, chest_lid)
        chest_label = Text("Digital Chest", font_size=18, color=WHITE)
        self.place_at_grid(chest, "D5")
        chest_label.next_to(chest, DOWN, buff=0.1)
        
        self.play(FadeIn(detective), Write(detective_label), FadeIn(chest), Write(chest_label))
        
        # Detective moves to lock chest
        self.play(
            detective.animate.move_to(self.grid["D4"]),
            detective_label.animate.next_to(self.grid["D4"], DOWN, buff=0.1)
        )
        
        # Locking action: a small padlock appears on the chest
        chest_lock = padlock_svg.copy().scale(0.4).move_to(chest.get_center())
        self.play(FadeIn(chest_lock))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A massive golden 'Mathematical Wall' (#FFD700) rises around the chest.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FFD700")
        
        # Massive golden wall (fortress representation)
        wall = RoundedRectangle(corner_radius=0.2, color="#FFD700", fill_opacity=0.05, stroke_width=12)
        wall.surround(VGroup(chest, chest_lock), buff=0.6)
        wall_label = Text("Mathematical Fortress", font_size=22, color="#FFD700")
        wall_label.next_to(wall, UP, buff=0.3)
        
        self.play(Create(wall), Write(wall_label))
        
        # Visual representation of "vastness" - filling the grid with faint padlock icons
        vastness = VGroup()
        bg_lock_template = padlock_svg.copy().set_opacity(0.15).scale(0.25)
        
        grid_spots = [
            "A1", "A2", "A3", "A4", "A5", "A6",
            "B1", "B2", "B3", "B4", "B5", "B6",
            "C1", "C2", "C3", "C4", "C5", "C6",
            "D1", "D3", "D6",
            "E1", "E2", "E3", "E4", "E5", "E6",
            "F1", "F2", "F3", "F4", "F5", "F6"
        ]
        
        for spot in grid_spots:
            lock_clone = bg_lock_template.copy()
            lock_clone.move_to(self.grid[spot])
            vastness.add(lock_clone)
            
        self.play(FadeIn(vastness))
        
        # Final glow to emphasize security
        self.play(Indicate(wall, color="#FFD700", scale_factor=1.1))
        self.wait(4)
