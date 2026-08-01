from manim import *
import numpy as np

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
        # Data
        title_text = "Conclusion: The Mathematical Wall"
        lecture_lines = [
            "SHA-256 is the gold standard for modern digital security.",
            "It is protected by the sheer physical impossibility of search.",
            "Your data stays safe behind this unbreakable mathematical wall."
        ]
        
        # Setup
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        BLUE_WALL = "#0000FF"
        RED_HACKER = "#FF0000"
        SILVER_VAULT = "#C0C0C0"
        WHITE_CAT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # A wall made of glowing blue hex codes (#0000FF) forms on screen.
        self.play(self.lecture[0].animate.set_color(BLUE_WALL))
        
        # Create a grid of hex codes representing the "wall"
        hex_data = [
            "A3C4", "1F2E", "9D8B", "F0A1",
            "7654", "E1F0", "A2B3", "B7C8",
            "C5D6", "E7F8", "0123", "D4E5",
            "9A8B", "7C6D", "5E4F", "3A2B"
        ]
        # Standard Text objects for stability (L022)
        hex_mobjects = VGroup(*[Text(h, font_size=20, color=BLUE_WALL) for h in hex_data])
        hex_mobjects.arrange_in_grid(rows=4, cols=4, buff=0.4)
        
        # Issue 46: Expand 'hex_mobjects' to 'C3'-'F6' (Line 80) at scale 0.8
        self.place_in_area(hex_mobjects, "C3", "F6", scale_factor=0.8)
        
        # Set z-index high so it's in front of the vault later
        hex_mobjects.set_z_index(10)
        
        # Use entry animation Write (L011)
        self.play(Write(hex_mobjects), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A red 'Hacker' icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hacker.svg] (#FF0000) bounces off the wall and shatters.
        self.play(self.lecture[1].animate.set_color(RED_HACKER))
        
        # Issue 46: Asset Integration: Use '/scratch/pawsey1357/jthen/Code2Video/assets/icon/hacker.svg' for the red hacker icon.
        hacker_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hacker.svg"
        hacker = SVGMobject(hacker_path)
        hacker.set_color(RED_HACKER)
        
        # Issue 46: Move 'hacker' to 'A2' (Line 95) to create a buffer from the hex wall.
        self.place_at_grid(hacker, "A2", scale_factor=0.6)
        
        self.play(FadeIn(hacker))
        
        # Move towards the wall (impact point near C3)
        wall_contact_point = self.grid["C3"]
        self.play(hacker.animate.move_to(wall_contact_point), run_time=1.5, rate_func=rate_functions.ease_in_sine)
        
        # Indicate impact using Indicate (L004)
        self.play(Indicate(hacker, color=RED_HACKER, scale_factor=1.3))
        
        # Shatter effect using small triangles
        shards = VGroup(*[
            Triangle(fill_opacity=1).scale(0.15).set_color(RED_HACKER) 
            for _ in range(8)
        ])
        shards.move_to(hacker.get_center())
        
        # Scatter shards in various directions to simulate shattering
        self.play(
            FadeOut(hacker),
            *[shard.animate.shift(np.array([np.random.uniform(-1.5, -0.5), np.random.uniform(-1, 1), 0])) for shard in shards],
            run_time=1
        )
        self.play(FadeOut(shards))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pixel the Cat (#FFFFFF) sleeps inside a silver vault (#C0C0C0) behind the wall.
        self.play(self.lecture[2].animate.set_color(WHITE_CAT))
        
        # Create Vault using RoundedRectangle (L009 fallback - no specific asset for vault/cat)
        vault_rect = RoundedRectangle(corner_radius=0.1, height=1.5, width=1.5, color=SILVER_VAULT, fill_opacity=0.3)
        vault_label = Text("VAULT", font_size=16, color=SILVER_VAULT).next_to(vault_rect, UP, buff=0.1)
        vault_group = VGroup(vault_rect, vault_label)
        
        # Create Cat representation using text (standard Text for stability)
        cat = Text("🐱", color=WHITE_CAT, font_size=32)
        
        # Set z-indices
        vault_group.set_z_index(2)
        cat.set_z_index(3)
        
        # Issue 46: Move 'vault_group' to 'B5' (Line 140) and 'cat' to 'B4' (Line 141)
        # Note: placing cat at B4 and vault at B5 separates them visually as requested by "keep them away from wall" 
        # and "legibility" concerns.
        self.place_at_grid(vault_group, "B5", scale_factor=0.8)
        self.place_at_grid(cat, "B4", scale_factor=0.7)
        
        # Fade in
        self.play(FadeIn(vault_group), FadeIn(cat))
        
        # Final emphasis: hex codes pulse to show security
        self.play(Indicate(hex_mobjects, color=BLUE_WALL, scale_factor=1.05), run_time=2)
        
        self.wait(3)
